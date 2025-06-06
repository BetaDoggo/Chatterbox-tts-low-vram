# Chatterbox with chunking for low vram
This is a gradio ui for the chatterbox tts library that supports chunking for low vram and longer generations. The chunks are created by grouping the input into sets of sentences, the chunks have their empty space removed before regrouping to make the result more cohesive. The UI will never split up a sentence, so very long run-on sentences might cause issues. Larger chunks should be more coherent, but I find as low as 200 is still pretty good. 500 or below is recommended on 6GB cards.

# Setup
1. Install the relevant version of pytorch: https://pytorch.org/
2. `pip install -r requirements.txt`
3. `python ui.py`
# Preview
![Screenshot 2025-06-06 184748](https://github.com/user-attachments/assets/247c2180-6a31-4f56-ab33-82f9e6bc059a)
