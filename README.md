# Discord Gemini Live (Native Audio) 🎙️✨

**The first open-source implementation of a Discord Bot utilizing the Google Gemini Multimodal Live API for native Speech-to-Speech interaction.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Gemini 2.0](https://img.shields.io/badge/Gemini-2.0_Flash_Live-magenta)](https://ai.google.dev/)
[![Discord.py](https://img.shields.io/badge/discord.py-2.0+-5865F2)](https://discordpy.readthedocs.io/)

<a href="https://github.com/sponsors/malvinarum">
  <img src="https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?style=for-the-badge&logo=github&logoColor=white" alt="Sponsor on GitHub" />
</a>
<a href="https://www.patreon.com/malvinarum">
  <img src="https://img.shields.io/badge/Patreon-Support-f96854?style=for-the-badge&logo=patreon&logoColor=white" alt="Support on Patreon" />
</a>
<a href="https://www.buymeacoffee.com/malvinarum">
  <img src="https://img.shields.io/badge/Buy_Me_A_Coffee-Donate-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy Me A Coffee" />
</a>

## 🚀 Why is this special?

Most "voice" bots on Discord today utilize a slow, chained pipeline:
`Speech-to-Text (Whisper)` ➔ `LLM (GPT)` ➔ `Text-to-Speech (ElevenLabs)`

**This bot is different.** It establishes a **direct, bi-directional WebSocket** connection with Google's Gemini 2.0 model.
* **No Transcriptions:** The model "hears" the raw audio bytes (tone, emotion, pace).
* **No TTS Engine:** The model generates raw audio bytes directly.
* **Sub-Second Latency:** Responses feel almost instantaneous.
* **Barge-In Capable:** You can interrupt the bot, and it will stop talking and listen (Echo Cancellation).

---

## 🛠️ The Architecture (The "Secret Sauce")

Connecting Discord's UDP audio stream to Gemini's WebSocket required solving several complex synchronization issues. This repo implements three critical fixes:

### 1. "Silence Injection" (Keep-Alive) 🤫
Gemini's WebSocket will close the connection with a `1011` error if the client stops sending data. However, when the bot is speaking, we must cut the microphone stream to prevent the bot from hearing itself (Echo).
* **Solution:** When the bot speaks, we inject **Digital Silence** (`b'\x00'`) into the upload stream. This "mutes" the mic but keeps the WebSocket heartbeat alive.

### 2. Accumulation Buffer (Jitter Fix) 🌊
Discord sends audio in tiny 20ms chunks. Sending these individually to Google causes network congestion and "choppy" audio.
* **Solution:** We implement an **Accumulation Buffer** that collects ~150ms of audio (4800 bytes) before sending a single, stable chunk to the API.

### 3. Opus Error Patching 🩹
Discord occasionally sends empty or malformed Opus packets, which causes standard decoders to crash.
* **Solution:** A monkey-patch for `discord.opus.Decoder` that safely returns silence instead of raising an exception.

---

## ⚙️ Prerequisites

* **Python 3.10+**
* **FFmpeg** (Required for Discord audio processing)
    * *Linux:* `sudo apt install ffmpeg`
    * *Windows:* [Download and add to PATH](https://ffmpeg.org/download.html)
    * *Mac:* `brew install ffmpeg`
* **Google Gemini API Key** (Access to `gemini-2.0-flash-exp` or newer)

---

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/discord-gemini-live.git](https://github.com/yourusername/discord-gemini-live.git)
    cd discord-gemini-live
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create your `.env` file:**
    Copy `.env.example` to `.env` and fill in your details.
    ```bash
    cp .env.example .env
    ```

---

## 📝 Configuration (`.env`)

| Variable | Description |
| :--- | :--- |
| `DISCORD_TOKEN` | Your Discord Bot Token (Get it from Developer Portal). |
| `GEMINI_API_KEY` | Your Google AI Studio API Key. |
| `GEMINI_MODEL_ID` | Default: `gemini-2.5-flash-native-audio-preview-12-2025` |
| `GEMINI_VOICE_NAME` | Voices: `Aoede`, `Puck`, `Charon`, `Kore`, `Fenrir`. |
| `BOT_PERSONALITY` | The System Instruction (Prompt) for the bot. |

**Example Personality:**
```text
You are Skippy, a grumpy otter wizard who hates technology but loves fish.
