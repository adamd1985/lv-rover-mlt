import re
import PIL.Image
from transformers import AutoProcessor, VisionEncoderDecoderModel

class CompetitionTranscriber:

    def __init__(self) -> None:
        self.device = 'cuda' # Set to 'cpu' if not using GPUs.
        self.processor = AutoProcessor.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2', use_fast=False)
        self.model = VisionEncoderDecoderModel.from_pretrained('naver-clova-ix/donut-base-finetuned-cord-v2')
        self.model.to(self.device)

        task_prompt = '<s_cord-v2>'
        self.decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)

    def transcribe(self, image: PIL.Image) -> str:
        pixel_values = self.processor(image, return_tensors='pt').pixel_values.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=self.decoder_input_ids,
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )[0]
        text = self.processor.batch_decode(outputs)[0]
        text = re.sub('<[^>]*>', '', text)
        text = re.sub('  +', ' ', text)
        return text
