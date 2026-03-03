import random
from typing import List, Dict, Any
from vetinari.subtask_tree import subtask_tree


AGENT_CAPABILITIES = {
    "explorer": ["research", "survey", "explore", "discover"],
    "librarian": ["document", "write", "organize", "catalog"],
    "oracle": ["design", "architect", "plan", "strategy"],
    "ui_planner": ["ui", "design", "interface", "ux"],
    "builder": ["implement", "build", "code", "create"],
    "researcher": ["research", "analyze", "study", "investigate"],
    "evaluator": ["test", "evaluate", "validate", "review"],
    "synthesizer": ["synthesize", "combine", "summarize", "integrate"]
}


def score_agent_for_subtask(agent_type: str, subtask: dict) -> float:
    score = 0
    subtask_desc = subtask.get('description', '').lower()
    subtask_prompt = subtask.get('prompt', '').lower()
    subtask_text = subtask_desc + " " + subtask_prompt
    
    agent_caps = AGENT_CAPABILITIES.get(agent_type, [])
    
    for cap in agent_caps:
        if cap in subtask_text:
            score += 30
    
    if subtask.get('agent_type') == agent_type:
        score += 20
    
    depth = subtask.get('depth', 0)
    if depth > 8:
        if agent_type in ["oracle", "researcher"]:
            score += 10
    elif depth < 3:
        if agent_type in ["builder", "evaluator"]:
            score += 10
    
    estimated_effort = subtask.get('estimated_effort', 1.0)
    if estimated_effort > 2.0:
        if agent_type in ["oracle", "builder"]:
            score += 15
    
    return score


def assign_subtasks(plan_id: str, auto_assign: bool = True) -> Dict:
    all_subtasks = subtask_tree.get_all_subtasks(plan_id)
    
    if not all_subtasks:
        return {
            "plan_id": plan_id,
            "assigned": 0,
            "unassigned": 0,
            "subtasks": []
        }
    
    agent_types = list(AGENT_CAPABILITIES.keys())
    assignments = []
    
    for subtask in all_subtasks:
        if subtask.assigned_agent and not auto_assign:
            continue
        
        scores = []
        for agent in agent_types:
            agent_score = score_agent_for_subtask(agent, subtask.to_dict())
            scores.append((agent, agent_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if scores and scores[0][1] > 0:
            best_agent = scores[0][0]
            subtask_tree.update_subtask(plan_id, subtask.subtask_id, {
                'assigned_agent': best_agent,
                'status': 'assigned'
            })
            assignments.append({
                'subtask_id': subtask.subtask_id,
                'assigned_agent': best_agent,
                'score': scores[0][1]
            })
        else:
            assignments.append({
                'subtask_id': subtask.subtask_id,
                'assigned_agent': '',
                'score': 0,
                'reason': 'no suitable agent found'
            })
    
    assigned_count = sum(1 for a in assignments if a['assigned_agent'])
    unassigned_count = len(assignments) - assigned_count
    
    return {
        "plan_id": plan_id,
        "assigned": assigned_count,
        "unassigned": unassigned_count,
        "assignments": assignments
    }


def execute_assignment_pass(plan_id: str, auto_assign: bool = True) -> Dict:
    result = assign_subtasks(plan_id, auto_assign)
    return result
