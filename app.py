"""
app.py - Gradio UI for the lean KOA X-ray screening cascade.

Wraps predict.py. Upload a knee X-ray -> get a screening verdict + the raw output.
A medical disclaimer is shown at the top and bottom of the page at all times.
"""

import gradio as gr
from predict import predict, DISCLAIMER, GRADE_KEY, DISEASED_THRESHOLD, SEVERE_THRESHOLD

TITLE = "Knee Osteoarthritis X-ray Screening (Educational Demo)"

INTRO = (
    "Upload a **knee X-ray** image. This demo runs a two-stage screening cascade: "
    "first it decides Healthy vs Diseased, and if Diseased it then estimates "
    "Moderate vs Severe. It serves a single ResNet50V2 per stage for speed."
)


def run(image):
    """Gradio callback: returns (human-readable summary, full result dict)."""
    if image is None:
        return "Please upload a knee X-ray image first.", {}

    result = predict(image)

    if result.get("rejected"):
        summary = (
            "NOT AN X-RAY - no screening was run.\n\n"
            f"{result['reason']}\n\n"
            f"(grayscale check: channel_saturation = {result['channel_saturation']}, "
            "an X-ray should be near 0)"
        )
        return summary, result

    if "error" in result:
        return result["error"], result

    verdict = result["result"]
    conf = result.get("confidence", 0.0)

    if verdict == "Healthy":
        stage_note = (
            f"Diseased probability {result['p_diseased']:.1%} is below the "
            f"{DISEASED_THRESHOLD:.0%} screening threshold."
        )
    else:
        stage_note = (
            f"Diseased probability {result['p_diseased']:.1%} cleared the "
            f"{DISEASED_THRESHOLD:.0%} threshold, so the specialist ran. "
            f"Severe probability {result.get('p_severe', 0):.1%} "
            f"(threshold {SEVERE_THRESHOLD:.0%})."
        )

    summary = (
        f"Screening result: {verdict}\n"
        f"Confidence: {conf:.1%}\n\n"
        f"{stage_note}\n\n"
        f"{GRADE_KEY}\n\n"
        "Reminder: educational screening only, not a diagnosis."
    )
    return summary, result


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DISCLAIMER)
    gr.Markdown(INTRO)
    gr.Markdown(f"_{GRADE_KEY}_")

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="numpy", label="Knee X-ray image")
            analyze_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            summary_out = gr.Textbox(label="Screening result", lines=8)
            json_out = gr.JSON(label="Full model output")

    analyze_btn.click(fn=run, inputs=image_in, outputs=[summary_out, json_out])

    gr.Markdown(DISCLAIMER)

if __name__ == "__main__":
    # queue() serializes requests so two heavy inferences don't run at once on a
    # free CPU. On Hugging Face, do NOT pass share=True - the Space is already public.
    demo.queue().launch()