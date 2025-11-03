from PIL import Image

img1 = Image.open("Admission Demographic + Serial Transcriptomic.png")

img1.save("Serial Transcriptomic Venn Diagram.tiff", format='tiff', dpi=(300, 300))

img2 = Image.open("Transcriptomic Venn Diagram.png")

img2.save("Transcriptomic Venn Diagram.tiff", format='tiff', dpi=(300, 300))