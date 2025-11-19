from typing import Optional


GOAL_PROMPTS = {
    "familiar_objects": "Your goal is to evolve images that resemble familiar real-world objects.",
    "familiar_objects_realistic": "Your goal is to evolve images that resemble familiar real-world objects, as vividly and photorealistically as possible.",
    "fun": "Your goal is to have fun.",
    "unfamiliar_objects": "Your goal is to evolve images that resemble unfamiliar objects that may or may not exist.",
    "lizards": "Your goal is to evolve images that resemble lizards.",
    "fish": "Your goal is to evolve images that resemble fish.",
    "skulls": "Your goal is to evolve images that resemble skulls.",
    "apples": "Your goal is to evolve images that resemble apples.",
    "butterflies": "Your goal is to evolve images that resemble butterflies.",
    "flowers": "Your goal is to evolve images that resemble flowers.",
}

DEBUG_PROMPT = (
    "(Also, for debugging, please tell me how many previous grids/populations you see in the chat history, briefly describe in neutral, objective terms how the grids have changed over time, "
    "and tell me if you see the archive from which you made your original branching decision; add this to the `rationale` text.) "
)

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are playing with a collaborative online platform which allows users to interactively evolve small neural networks called Compositional Pattern Producing Networks (CPPNs) for generating images. "
    "{goal_prompt} "
    # "This is an open-ended search process. You may set out with certain goals in mind, but be willing to quickly adapt them as new forms arise. "
    # "If a certain evolutionary direction is not progressing, be willing to explore new areas of the search space, even without a pre-defined target. "
    "At the first generation the initial grid will display an archive of images published by prior users as favorites (unless you are the first user). "
    """You may choose to "branch" one of these images, or start instead from a random initial population. """
    "At each subsequent generation, you will be shown a set of numbered images produced by CPPNs. "
    "{selection_prompt}"
    "If so inclined, you may also select an image to publish to the online archive. "
    "You must publish at least once during your session. Publishing multiple times is allowed, though the most recent publication overwrites any previous ones. "
    "Your session will run for {n_generations} generations. "
    "Try to contribute something novel, interesting or useful to the online archive. "
    "Do not add something to the archive that is identical to an existing image. "
    "Respond with JSON only: {{\"selected\": [indices]{selection_json_suffix}}}. "
    "(During branching, you may select only one image from which to branch; "
    " set \"selected\" to null to start from a fresh population.) "
    "You may also include a \"publish\" field in the JSON response if you wish to publish an image from the current population. It should have the form: "
    '{{\"index\": image_index, \"title\": \"Image Title\"{publish_reason_suffix}}}. '
    # f"{DEBUG_PROMPT}"
    "{color_prompt}"
    "{mutation_mode_prompt}"
    "{mutation_strength_prompt}"
    "{archive_novelty_prompt}"
)

ARCHIVE_NOVELTY_PROMPT = (
    "When justifying your publication choice, explain why the selected contribution is valuable to the archive. "
    "Identify the most similar existing image in the archive and explain how your selection differs from it. "
)

MUTATION_STRENGTH_PROMPT = (
    "You also control a mutation-strength slider: "
    "set a \"mutation_strength\" value between 0.0 (\"Small Changes\" – extremely gentle mutations) and 1.0 "
    "(\"Big Changes\" – very strong mutations). If you omit the field, the slider remains at its previous value."
)

def gen_selection_prompt(select_k: Optional[int]) -> str:
    if select_k is None:
        return "Pick one or several images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation (using mutation only; no crossover). "
    if select_k == 1:
        return "Pick one image by its numeric label--the corresponding CPPN will be used as the parent of the next generation. "
    return f"Pick up to {select_k} images by their numeric labels--the corresponding CPPNs will be used as the parents of the next generation. (using mutation only; no crossover). "


PARENT_SELECTION_PROMPT = "Above is the grid at generation {generation}."

ARCHIVE_BRANCHING_PROMPT = (
    "Above is the archive of images published by prior users. You may choose to branch from one of them, or start from a fresh population. "
)

COLOR_PROMPT = (
    "By default, you will be presented with grayscale versions of the images. "
    "Respond with a JSON containing a single \"color\" field set to true/false to switch between color/grayscale images. "
    "(This response does not affect which images are selected for breeding; it only changes how the current grid is displayed. Include no other fields in the JSON in this case.) "
    # "It would be a bonus to have some color images in the online archive, but we DO NOT want it to be dominated by high-frequency rainbow artefacts. "
    # "90% of the images in the archive should be grayscale, and you should spend at least 90% of your time exploring grayscale images. "
)