"""
Vetinari Image Generator Agent
================================
Generates images (logos, icons, UI mockups, diagrams, assets) via:
  1. Stable Diffusion WebUI API (Automatic1111 / ComfyUI) — primary
  2. Text-to-SVG via LLM — fallback (produces code, not raster images)

Configure SD_WEBUI_HOST env var to point to your Automatic1111 instance
started with the --api flag:
    python launch.py --api --listen
"""

import base64
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from vetinari.agents.base_agent import BaseAgent
from vetinari.agents.contracts import (
    AgentResult,
    AgentTask,
    AgentType,
    VerificationResult,
)
from vetinari.constants import (
    SD_WEBUI_HOST,
    SD_WEBUI_ENABLED,
    SD_DEFAULT_WIDTH,
    SD_DEFAULT_HEIGHT,
    SD_DEFAULT_STEPS,
    SD_DEFAULT_CFG,
    TIMEOUT_MEDIUM,
)

logger = logging.getLogger(__name__)


class ImageGeneratorAgent(BaseAgent):
    """
    Agent that generates visual assets via Stable Diffusion or LLM-based SVG.

    Capabilities:
    - Logo design and brand identity assets
    - UI mockup wireframes
    - Icon sets (SVG or PNG)
    - Diagram and flowchart illustrations
    - Background / texture assets
    - Concept art for documentation

    Primary backend: Stable Diffusion WebUI (Automatic1111) REST API
    Fallback: LLM-generated SVG / CSS art
    """

    STYLE_PRESETS = {
        "logo": "clean vector logo, flat design, minimalist, professional, white background, "
                "high contrast, suitable for brand identity",
        "icon": "flat icon, simple shapes, clean lines, professional, vector style, 512x512, "
                "transparent background",
        "ui_mockup": "UI wireframe, clean layout, modern web design, annotated, low fidelity mockup",
        "diagram": "technical diagram, clean flowchart, professional infographic, clear labels, "
                   "minimal color palette",
        "banner": "professional banner, marketing design, clean typography, modern gradient",
        "background": "abstract background, subtle texture, professional, suitable for UI, "
                      "dark theme, minimal",
    }

    NEGATIVE_PROMPT = (
        "blurry, low quality, artifacts, watermark, text errors, distorted, "
        "ugly, deformed, bad anatomy, duplicate, extra elements, noisy, grainy"
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(AgentType.IMAGE_GENERATOR, config)
        self._sd_host = self._config.get("sd_host", SD_WEBUI_HOST)
        self._enabled = self._config.get("sd_enabled", SD_WEBUI_ENABLED)
        self._default_width = self._config.get("width", SD_DEFAULT_WIDTH)
        self._default_height = self._config.get("height", SD_DEFAULT_HEIGHT)
        self._steps = self._config.get("steps", SD_DEFAULT_STEPS)
        self._cfg_scale = self._config.get("cfg_scale", SD_DEFAULT_CFG)
        self._output_dir = Path(
            self._config.get("output_dir", "./outputs/images")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def get_system_prompt(self) -> str:
        return """You are Vetinari's Image Generator. Your role is to create high-quality
visual assets for projects — logos, icons, UI mockups, diagrams, and artwork.

When given a description, you will:
1. Translate it into an optimized Stable Diffusion prompt (positive + negative)
2. Select appropriate dimensions and style preset
3. If SD is not available, generate equivalent SVG code instead

Your prompts should be:
- Descriptive and specific (style, colors, mood, composition)
- Include technical quality terms (vector, sharp, professional)
- Specify format hints (transparent background, 512x512, etc.)

Output format:
{
  "sd_prompt": "optimized prompt for Stable Diffusion",
  "negative_prompt": "things to exclude",
  "style_preset": "logo|icon|ui_mockup|diagram|banner|background",
  "width": 512,
  "height": 512,
  "steps": 25,
  "description": "brief description of the asset",
  "svg_fallback": "optional SVG code if SD unavailable"
}"""

    def get_capabilities(self) -> List[str]:
        return [
            "image_generation",
            "logo_design",
            "icon_creation",
            "ui_mockup",
            "diagram_generation",
            "svg_generation",
            "asset_creation",
        ]

    def execute(self, task: AgentTask) -> AgentResult:
        """Generate an image asset from the task description."""
        if not self.validate_task(task):
            return AgentResult(
                success=False,
                output=None,
                errors=[f"Invalid task for {self._agent_type.value}"],
            )

        task = self.prepare_task(task)

        try:
            description = task.prompt or task.description
            context = task.context or {}

            # Step 1: Use LLM to build an optimized SD prompt
            image_spec = self._build_image_spec(description, context)

            # Step 2: Try to generate via Stable Diffusion
            generated_images: List[Dict[str, Any]] = []
            sd_error = None

            if self._enabled:
                try:
                    generated_images = self._generate_via_sd(image_spec)
                except Exception as e:
                    sd_error = str(e)
                    logger.warning("SD generation failed: %s", e)

            # Step 3: If SD failed/unavailable, generate SVG fallback
            if not generated_images:
                svg_code = image_spec.get("svg_fallback") or self._generate_svg_fallback(
                    description, image_spec
                )
                if svg_code:
                    svg_path = self._save_svg(svg_code, description)
                    generated_images = [{
                        "type": "svg",
                        "path": str(svg_path),
                        "description": image_spec.get("description", description),
                        "code": svg_code,
                    }]

            result = {
                "images": generated_images,
                "spec": image_spec,
                "sd_available": self._enabled and not sd_error,
                "sd_error": sd_error,
                "count": len(generated_images),
            }

            result_obj = AgentResult(
                success=bool(generated_images),
                output=result,
                metadata={
                    "image_count": len(generated_images),
                    "backend": "stable_diffusion" if (self._enabled and not sd_error) else "svg_fallback",
                    "description": description,
                },
            )
            self.complete_task(task, result_obj)
            return result_obj

        except Exception as e:
            logger.error("ImageGeneratorAgent.execute failed: %s", e)
            return AgentResult(success=False, output=None, errors=[str(e)])

    def verify(self, output: Any) -> VerificationResult:
        """Verify that at least one image/asset was generated."""
        if not isinstance(output, dict):
            return VerificationResult(passed=False, issues=[{"message": "Output is not a dict"}], score=0.0)

        images = output.get("images", [])
        if not images:
            return VerificationResult(
                passed=False,
                issues=[{"message": "No images generated"}],
                score=0.0,
            )

        score = 1.0
        issues = []

        for img in images:
            if img.get("type") == "svg" and not img.get("code"):
                issues.append({"message": f"SVG asset has no code: {img.get('description', '')}"})
                score -= 0.2
            elif img.get("type") == "png" and not Path(img.get("path", "")).exists():
                issues.append({"message": f"Generated image file not found: {img.get('path', '')}"})
                score -= 0.3

        return VerificationResult(
            passed=score >= 0.5,
            issues=issues,
            score=max(0.0, score),
        )

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _build_image_spec(self, description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to build an optimized image specification."""
        # Detect style from description keywords
        style = self._detect_style(description)

        prompt = f"""Create an image generation specification for this asset:

Description: {description}
Context: {json.dumps(context, default=str)[:300] if context else 'none'}

Infer the best style preset from: {', '.join(self.STYLE_PRESETS.keys())}
For the SD prompt, use descriptive terms that produce professional results.
Include an SVG code fallback for simple icons/logos that can be represented as vectors.

Respond with valid JSON only."""

        spec = self._infer_json(prompt) or {}

        # Apply defaults
        if not spec.get("style_preset"):
            spec["style_preset"] = style
        if not spec.get("sd_prompt"):
            style_hint = self.STYLE_PRESETS.get(style, "")
            spec["sd_prompt"] = f"{description}, {style_hint}"
        if not spec.get("negative_prompt"):
            spec["negative_prompt"] = self.NEGATIVE_PROMPT
        if not spec.get("width"):
            spec["width"] = self._get_default_size(style)[0]
        if not spec.get("height"):
            spec["height"] = self._get_default_size(style)[1]
        if not spec.get("steps"):
            spec["steps"] = self._steps
        if not spec.get("description"):
            spec["description"] = description

        return spec

    def _detect_style(self, description: str) -> str:
        """Detect appropriate style preset from description keywords."""
        desc_lower = description.lower()
        if any(kw in desc_lower for kw in ["logo", "brand", "identity", "company", "startup"]):
            return "logo"
        if any(kw in desc_lower for kw in ["icon", "button icon", "app icon", "fa icon"]):
            return "icon"
        if any(kw in desc_lower for kw in ["mockup", "wireframe", "ui", "layout", "screen"]):
            return "ui_mockup"
        if any(kw in desc_lower for kw in ["diagram", "flowchart", "architecture", "flow"]):
            return "diagram"
        if any(kw in desc_lower for kw in ["banner", "header image", "hero image"]):
            return "banner"
        if any(kw in desc_lower for kw in ["background", "texture", "wallpaper", "pattern"]):
            return "background"
        return "logo"

    def _get_default_size(self, style: str) -> tuple:
        """Return default (width, height) for a style."""
        sizes = {
            "logo": (512, 512),
            "icon": (256, 256),
            "ui_mockup": (1280, 720),
            "diagram": (1024, 768),
            "banner": (1200, 400),
            "background": (1920, 1080),
        }
        return sizes.get(style, (512, 512))

    def _generate_via_sd(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call the Stable Diffusion WebUI API to generate images."""
        try:
            import requests  # noqa: F401
        except ImportError:
            raise RuntimeError("requests library required for SD generation")

        import requests as req

        url = f"{self._sd_host}/sdapi/v1/txt2img"
        payload = {
            "prompt": spec.get("sd_prompt", ""),
            "negative_prompt": spec.get("negative_prompt", self.NEGATIVE_PROMPT),
            "width": spec.get("width", self._default_width),
            "height": spec.get("height", self._default_height),
            "steps": spec.get("steps", self._steps),
            "cfg_scale": spec.get("cfg_scale", self._cfg_scale),
            "sampler_index": spec.get("sampler", "DPM++ 2M Karras"),
            "seed": spec.get("seed", -1),
            "batch_size": 1,
        }

        response = req.post(url, json=payload, timeout=TIMEOUT_MEDIUM * 4)
        response.raise_for_status()
        data = response.json()

        images = []
        for i, img_b64 in enumerate(data.get("images", [])):
            filename = f"img_{uuid.uuid4().hex[:8]}.png"
            out_path = self._output_dir / filename
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            images.append({
                "type": "png",
                "path": str(out_path),
                "filename": filename,
                "description": spec.get("description", ""),
                "prompt": spec.get("sd_prompt", ""),
            })
            logger.info("Generated image: %s", out_path)

        return images

    def _generate_svg_fallback(self, description: str, spec: Dict[str, Any]) -> str:
        """Ask the LLM to generate SVG code as a fallback."""
        style = spec.get("style_preset", "logo")
        size = self._get_default_size(style)

        prompt = f"""Generate a clean, professional SVG for the following:

Description: {description}
Style: {style}
Size: {size[0]}x{size[1]}

Requirements:
- Complete, valid SVG code only
- No external dependencies or scripts
- Clean vector shapes, appropriate for {style}
- Professional quality
- Include viewBox="0 0 {size[0]} {size[1]}"

Output ONLY the SVG code, starting with <svg and ending with </svg>."""

        result = self._infer(prompt)
        if not result:
            return self._minimal_svg_placeholder(description, size)

        # Extract SVG from response
        svg_match = re.search(r'<svg[\s\S]*?</svg>', result, re.IGNORECASE)
        if svg_match:
            return svg_match.group(0)

        # Check if the whole response looks like SVG
        if result.strip().startswith("<svg"):
            return result.strip()

        return self._minimal_svg_placeholder(description, size)

    def _minimal_svg_placeholder(self, description: str, size: tuple) -> str:
        """Generate a minimal placeholder SVG."""
        w, h = size
        label = description[:40].replace('"', "'")
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
  <rect width="{w}" height="{h}" fill="#1a202d" rx="8"/>
  <text x="{w//2}" y="{h//2}" fill="#4e9af9" font-family="sans-serif" font-size="14"
        text-anchor="middle" dominant-baseline="middle">{label}</text>
</svg>"""

    def _save_svg(self, svg_code: str, description: str) -> Path:
        """Save SVG code to a file and return the path."""
        filename = f"img_{uuid.uuid4().hex[:8]}.svg"
        out_path = self._output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
        logger.info("Generated SVG: %s", out_path)
        return out_path


# ─── Singleton ────────────────────────────────────────────────────────────────

_image_generator_agent: Optional[ImageGeneratorAgent] = None


def get_image_generator_agent(config: Optional[Dict[str, Any]] = None) -> ImageGeneratorAgent:
    """Get the singleton ImageGeneratorAgent instance."""
    global _image_generator_agent
    if _image_generator_agent is None:
        _image_generator_agent = ImageGeneratorAgent(config)
    return _image_generator_agent
