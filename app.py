from fastai.vision import *
from fastapi import FastAPI, UploadFile, File, Query, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import io, time

learn = load_learner('.')

def predictions(img):
    return learn.predict(img)[1]

def redact(img, mask):
    # img: m x n, mask: i x j
    resized_mask = np.array(PIL.Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size)) > 0
    redacted = np.array(img)
    redacted[resized_mask] *= 0
    return PIL.Image.fromarray(redacted)

def mask_categories(img, categories):
    start_time = time.time()
    segs = predictions(img).numpy()[0].astype(np.uint8)
    end_time = time.time()
    print(f"prediction took {end_time - start_time}")
    empty = segs < 0
    masked = empty
    for c in categories:
      masked |= segs == c
    return masked

def redact_categories(rgbimg, rgbaimg, categories):
    masked = mask_categories(rgbimg, categories)
    return redact(rgbaimg, masked)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"app-name": "camvid-segmentation"}

@app.get("/health")
def read_health():
    return {"status": "hale and hearty and limber and spry"}

@app.post("/mask")
async def read_mask(image: UploadFile = File(...), categories: str = Query("")):
    rgbimg = open_image(image.file, convert_mode='RGB')
    intcats = [int(c) for c in categories.split(',')]
    imgio = io.BytesIO()
    PIL.Image.fromarray(255-mask_categories(rgbimg, intcats).astype(np.uint8)*255).save(imgio, "PNG")
    imgio.seek(0)
    return Response(content=imgio.read(), media_type="image/png")

@app.post("/redact")
async def read_redact(image: UploadFile = File(...), categories: str = Query("")):
    rgbaimg = PIL.Image.open(image.file).convert('RGBA')
    rgbimg = open_image(image.file, convert_mode='RGB')
    intcats = [int(c) for c in categories.split(',')]
    imgio = io.BytesIO()
    redact_categories(rgbimg, rgbaimg, intcats).save(imgio, "PNG")
    imgio.seek(0)
    return Response(content=imgio.read(), media_type="image/png")
