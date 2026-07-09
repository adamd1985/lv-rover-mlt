import subprocess
import tempfile
import os
import PIL.Image
import malti.line_joiner

# This assumes that this version of Tesseract has been installed: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe
# and that the mlt.traineddata Tesseract model has been added to C:\Program Files\Tesseract-OCR\tessdata: https://github.com/tesseract-ocr/tessdata_best/blob/main/mlt.traineddata
# To use NOMOCRAT fine-tuned Tesseract, replace mlt.traineddata with https://github.com/vanyagelfo/NOMOCRAT-OCR/blob/main/tessdata_custom/mlt_custom_v1.traineddata (rename it to 'mlt.trainedata').
# It also assumes that malti==0.3.1 has been installed for the line joiner algorithm.
class CompetitionTranscriber:

    def __init__(self) -> None:
        self.line_joiner = malti.line_joiner.RBLineJoiner()

    def transcribe(self, image: PIL.Image) -> str:
        with tempfile.TemporaryDirectory() as path:
            image.save(os.path.join(path, 'img.jpg'))
            subprocess.run(
                [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    '-l', 'mlt',
                    os.path.join(path, 'img.jpg'),
                    os.path.join(path, 'out'),
                ],
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with open(os.path.join(path, 'out.txt'), encoding='utf-8') as f:
                text = self.line_joiner.join_lines(f.read().strip().split('\n'), fix_hyphenated_words=True)
        return text
