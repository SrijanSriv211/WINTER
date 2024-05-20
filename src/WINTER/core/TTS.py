# https://github.com/rany2/edge-tts.git
from ..shared.utils import get_bin_path
import playsound, asyncio, edge_tts, os

# Speak out loud the text
def Speak(text, voice="en-US-GuyNeural"):
    cache_path = os.path.join(get_bin_path(), "cache")
    output = os.path.join(cache_path, "audio.mp3")
    if not os.path.isdir(cache_path):
        os.mkdir(cache_path)

    # TTS engine
    async def _main(text, voice):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_main(text, voice))

    print(text)
    playsound.playsound(output)
    os.remove(output)
