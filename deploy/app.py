import argparse

import uvicorn
from comma_fixer import CommaFixer
from fastapi import FastAPI
from logger import logger as base_loger
from params import config_path
from typings import FixedText, InputText

logger = base_loger.bind(corr_id="MAIN ")

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=80, help="Specify port to run the service.")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

app = FastAPI()
comma_fixer = None


@app.post("/", response_model=FixedText, status_code=200)
def fix_commas(data: InputText):
    input_text = data.input_text
    logger.debug(f"Got the incomming text: {input_text}")
    text_with_commas = comma_fixer.fix_commas(input_text)
    logger.debug(f"Model response: {text_with_commas}")
    return {"text_with_commas": text_with_commas, "original_text": input_text}


if __name__ == "__main__":
    logger.info("Booting up a CommaFixer service...")
    comma_fixer = CommaFixer(config_path, args.device)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
