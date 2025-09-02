# 🎤 Rhyme Rarity Checker

**Rhyme Rarity Checker** is an AI-powered tool that evaluates how well two words rhyme and how rare that rhyme pairing is in published literature, music, and poetry. Built with **Gradio** and powered by **OpenAI's GPT-3.5-turbo**, it distinguishes between rhyme types such as *perfect, slant, assonance, consonance*, and *forced*.

👉 **Live Demo** on Hugging Face Spaces: [Try it here]([https://huggingface.co/spaces/your-username/RhymeRater)

---

## ✨ Features

- 🎯 **Rhyme Type Classification**  
  Detects rhyme types: perfect, slant, assonance, consonance, forced, or none.

- 📈 **Rarity Score (0–100)**  
  Indicates how rare the rhyme pairing is in published use (0 = common, 100 = rare).

- 🧠 **LLM-Powered Explanation**  
  Provides a brief explanation of the rhyme logic used and reference examples.

- 🔧 **Gradio UI**  
  Clean, responsive, and easy to use for poets, lyricists, rappers, and creatives.

---

## 🧪 Example Pairs

| Word 1   | Word 2      | Rhyme Type | Notes                              |
|----------|-------------|-------------|-------------------------------------|
| cat      | hat         | Perfect     | Common children's rhyme             |
| orange   | door hinge  | Forced      | Popular example of a near-rhyme     |
| fire     | choir       | Slant       | Similar vowels, different consonants |
| love     | above       | Slant       | Frequently used but imperfect       |

---

## 🚀 Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rhyme-rarity-checker.git
cd rhyme-rarity-checker
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Create a `.env` file:

```env
OPENAI_API_KEY=sk-your-key-here
```

Or set the environment variable manually.

### 4. Run the app

```bash
python app.py
```

It will launch on [http://localhost:7860](http://localhost:7860)

---

## 📦 Requirements

From `requirements.txt`:

```
openai==0.28.1
gradio
```

---

## 🔐 API Key Info

To use this app, you must have:
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- A payment method enabled (ChatGPT Plus does *not* include API credits)

---

## 🤝 Contributing

Want to suggest rhyme logic improvements, rhyme corpus integration, or add offline fallback?  
Feel free to fork and open a pull request!

---

## 📄 License

MIT License

---

## 🙌 Credits

Built by [your-name] using:
- [Gradio](https://gradio.app)
- [OpenAI API](https://platform.openai.com)
- [Hugging Face Spaces](https://huggingface.co/spaces)
