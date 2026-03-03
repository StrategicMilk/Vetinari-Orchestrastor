import random
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from vetinari.template_loader import TemplateLoader
from vetinari.subtask_tree import SubtaskTree, subtask_tree
from vetinari.planning import Plan, PlanManager


DEFAULT_MAX_DEPTH = 14
MIN_MAX_DEPTH = 12
MAX_MAX_DEPTH = 16

SEED_MIX = {
    "oracle": 0.50,
    "researcher": 0.25,
    "explorer": 0.25
}

SEED_RATE = {
    "base": 2,
    "max": 4,
    "subtask_min": 20,
    "subtask_max": 40
}

RECURSION_KNOBS = {
    "est_effort_threshold": 2.0,
    "inputs_threshold": 3,
    "outputs_threshold": 3,
    "complexity_keywords": ["design", "architect", "specify", "integrate", "architecture", "complex"]
}


class DecompositionAgent:
    def __init__(self):
        self.template_loader = TemplateLoader()
        self.subtask_tree = subtask_tree

    def get_effective_max_depth(self, plan: Plan, subtask_override: int = 0) -> int:
        if subtask_override > 0:
            return max(MIN_MAX_DEPTH, min(MAX_MAX_DEPTH, subtask_override))
        if plan.max_depth_override > 0:
            return max(MIN_MAX_DEPTH, min(MAX_MAX_DEPTH, plan.max_depth_override))
        return DEFAULT_MAX_DEPTH

    def select_seeds(self, seed_rate: int = 2) -> List[str]:
        total_seeds = min(seed_rate, SEED_RATE["max"])
        seeds = []
        
        num_oracle = int(total_seeds * SEED_MIX["oracle"])
        num_researcher = int(total_seeds * SEED_MIX["researcher"])
        num_explorer = total_seeds - num_oracle - num_researcher
        
        seeds.extend(["oracle"] * num_oracle)
        seeds.extend(["researcher"] * num_researcher)
        seeds.extend(["explorer"] * num_explorer)
        
        random.shuffle(seeds)
        return seeds[:total_seeds]

    def should_decompose(self, depth: int, effective_max_depth: int, 
                        estimated_effort: float, inputs: List, outputs: List,
                        description: str) -> bool:
        if depth >= effective_max_depth:
            return False
        
        if estimated_effort > RECURSION_KNOBS["est_effort_threshold"]:
            return True
        
        if len(inputs) + len(outputs) > RECURSION_KNOBS["inputs_threshold"] + RECURSION_KNOBS["outputs_threshold"]:
            return True
        
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in RECURSION_KNOBS["complexity_keywords"]):
            return True
        
        return False

    def decompose(self, plan: Plan, parent_id: str = "root", 
                  depth: int = 0, prompt_context: str = None) -> List[Dict]:
        effective_max_depth = self.get_effective_max_depth(plan)
        
        if depth >= effective_max_depth:
            return []
        
        template_version = plan.template_version or "v1"
        all_templates = self.template_loader.load_templates(version=template_version)
        
        if not all_templates:
            return []
        
        keywords = []
        if prompt_context:
            keywords = prompt_context.lower().split()
        
        matching_templates = []
        for t in all_templates:
            t_keywords = t.get('keywords', [])
            if keywords:
                if any(kw in t_keywords for kw in keywords):
                    matching_templates.append(t)
            else:
                matching_templates.append(t)
        
        if not matching_templates:
            matching_templates = all_templates[:8]
        
        seeds = self.select_seeds(plan.seed_rate or SEED_RATE["base"])
        
        subtasks_created = []
        for i, template in enumerate(matching_templates[:SEED_RATE["subtask_max"]]):
            subtask = self.subtask_tree.create_subtask(
                plan_id=plan.plan_id,
                parent_id=parent_id,
                depth=depth,
                description=template.get('name', ''),
                prompt=template.get('prompt_template', '').format(input=prompt_context or ''),
                agent_type=template.get('agent_type', 'builder'),
                max_depth=effective_max_depth,
                max_depth_override=0,
                dod_level=template.get('dod_level', 'Standard'),
                dor_level=template.get('dor_level', 'Standard'),
                estimated_effort=template.get('estimated_effort', 1.0),
                inputs=template.get('inputs', []),
                outputs=template.get('outputs', []),
                decomposition_seed=seeds[i % len(seeds)] if seeds else ""
            )
            subtasks_created.append(subtask.to_dict())
            
            if self.should_decompose(depth, effective_max_depth, 
                                    template.get('estimated_effort', 1.0),
                                    template.get('inputs', []),
                                    template.get('outputs', []),
                                    template.get('description', '')):
                child_subtasks = self.decompose(
                    plan=plan,
                    parent_id=subtask.subtask_id,
                    depth=depth + 1,
                    prompt_context=template.get('prompt_template', '').format(input=prompt_context or '')
                )
                subtasks_created.extend(child_subtasks)
        
        return subtasks_created

    def decompose_from_prompt(self, plan: Plan, prompt: str) -> Dict:
        subtasks = self.decompose(plan=plan, parent_id="root", depth=0, prompt_context=prompt)
        
        return {
            "plan_id": plan.plan_id,
            "subtasks": subtasks,
            "count": len(subtasks),
            "depth": self.subtask_tree.get_tree_depth(plan.plan_id),
            "effective_max_depth": self.get_effective_max_depth(plan)
        }


decomposition_agent = DecompositionAgent()
