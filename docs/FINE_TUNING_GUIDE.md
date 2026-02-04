# Guide de Fine-Tuning Llama pour Viralify

Ce guide explique comment entraîner votre propre modèle Llama Instruct avec des données synthétiques pour réduire les coûts d'API.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Génération de données synthétiques](#génération-de-données-synthétiques)
3. [Préparation du dataset](#préparation-du-dataset)
4. [Fine-tuning avec Unsloth](#fine-tuning-avec-unsloth)
5. [Fine-tuning avec Axolotl](#fine-tuning-avec-axolotl)
6. [Déploiement du modèle](#déploiement-du-modèle)
7. [Intégration avec Viralify](#intégration-avec-viralify)
8. [Estimation des coûts](#estimation-des-coûts)

---

## Vue d'ensemble

### Pourquoi fine-tuner?

| Approche | Coût/1M tokens | Latence | Qualité |
|----------|----------------|---------|---------|
| GPT-4o | ~$5-15 | 2-5s | ⭐⭐⭐⭐⭐ |
| Groq (Llama 70B) | ~$0.70 | 0.5s | ⭐⭐⭐⭐ |
| Fine-tuned Llama 8B | ~$0.05* | 0.3s | ⭐⭐⭐⭐ |

*Coût d'inférence self-hosted (GPU A100 ~$1/h, ~20M tokens/h)

### Modèles recommandés

| Modèle | VRAM | Usage |
|--------|------|-------|
| **Llama-3.1-8B-Instruct** | 16GB | Génération de scripts, voiceovers |
| **Llama-3.1-70B-Instruct** | 140GB | Qualité maximale (multi-GPU) |
| **Mistral-7B-Instruct-v0.3** | 14GB | Alternative légère |

---

## Génération de données synthétiques

### Méthode 1: Utiliser le TrainingLogger existant

Viralify inclut déjà un système de collecte de données. Activez-le:

```bash
# Dans .env
TRAINING_LOGGER_ENABLED=true
TRAINING_DATA_PATH=/app/data/training_dataset.jsonl
```

Les interactions validées sont automatiquement enregistrées dans `/app/data/training_dataset.jsonl`.

### Méthode 2: Génération synthétique avec GPT-4

Créez un script pour générer des données synthétiques:

```python
# scripts/generate_synthetic_data.py

import json
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict
import random

client = AsyncOpenAI()

# Templates pour chaque type de tâche Viralify
TASK_TEMPLATES = {
    "presentation_script": {
        "system": """Tu es un expert en création de contenu éducatif.
Tu génères des scripts de présentation structurés en JSON.
Chaque slide doit avoir: title, content, voiceover, slide_type.""",
        "topics": [
            "Introduction à Python",
            "Architecture microservices",
            "Machine Learning basics",
            "Docker et Kubernetes",
            "API REST design",
            "Sécurité web OWASP",
            "Git et workflows",
            "Design patterns",
            # Ajouter 100+ topics variés
        ],
        "output_schema": {
            "slides": [
                {
                    "title": "string",
                    "content": ["bullet1", "bullet2"],
                    "voiceover": "string (50-100 mots)",
                    "slide_type": "title|content|code|diagram|conclusion"
                }
            ]
        }
    },
    "code_generation": {
        "system": """Tu es un expert en programmation.
Tu génères du code propre, commenté et fonctionnel.
Inclus toujours: code, explanation, key_concepts, expected_output.""",
        "topics": [
            "Implémenter une liste chaînée en Python",
            "Créer un serveur HTTP basique en Go",
            "Parser du JSON en Rust",
            "Connexion à PostgreSQL en Java",
            # Ajouter 100+ exercices de code
        ]
    },
    "quiz_generation": {
        "system": """Tu es un expert en évaluation pédagogique.
Tu crées des quiz style Udemy avec QCM et explications.""",
        "topics": [
            "Quiz sur les bases de Python",
            "Évaluation Docker niveau intermédiaire",
            # etc.
        ]
    }
}

async def generate_example(task_type: str, topic: str) -> Dict:
    """Génère un exemple d'entraînement pour une tâche donnée."""
    template = TASK_TEMPLATES[task_type]

    messages = [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": f"Génère du contenu pour: {topic}\n\nFormat JSON uniquement."}
    ]

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.7
    )

    return {
        "task_type": task_type,
        "messages": messages,
        "response": response.choices[0].message.content,
        "topic": topic
    }

async def generate_dataset(num_examples: int = 1000) -> List[Dict]:
    """Génère un dataset complet."""
    examples = []

    for task_type, template in TASK_TEMPLATES.items():
        topics = template["topics"]

        # Générer des variations pour chaque topic
        for topic in topics:
            # Variations de difficulté
            for difficulty in ["débutant", "intermédiaire", "avancé"]:
                varied_topic = f"{topic} (niveau {difficulty})"

                try:
                    example = await generate_example(task_type, varied_topic)
                    examples.append(example)
                    print(f"Generated: {task_type} - {varied_topic}")
                except Exception as e:
                    print(f"Error: {e}")

                # Rate limiting
                await asyncio.sleep(0.5)

    return examples

async def main():
    print("Generating synthetic training data...")
    examples = await generate_dataset(1000)

    # Sauvegarder en JSONL
    with open("synthetic_training_data.jsonl", "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} examples")

if __name__ == "__main__":
    asyncio.run(main())
```

### Méthode 3: Distillation depuis GPT-4

```python
# scripts/distill_from_gpt4.py

"""
Distillation: Utiliser GPT-4 pour générer des réponses de haute qualité,
puis entraîner un modèle plus petit sur ces réponses.
"""

import json
from openai import OpenAI

client = OpenAI()

def distill_presentation_generation():
    """Distille la capacité de génération de présentations."""

    # Prompts réels utilisés par Viralify
    prompts = [
        {
            "topic": "Introduction à Apache Kafka",
            "num_slides": 10,
            "target_audience": "développeurs seniors",
            "language": "fr"
        },
        # Ajouter 500+ variations
    ]

    training_examples = []

    for prompt in prompts:
        system_prompt = """Tu es un expert en création de présentations techniques.
Génère un script de présentation complet en JSON avec la structure:
{
    "title": "...",
    "slides": [
        {
            "title": "...",
            "slide_type": "title|content|code|diagram",
            "content": ["..."],
            "voiceover": "... (60-80 mots par slide)",
            "code": "... (si slide_type=code)",
            "diagram_description": "... (si slide_type=diagram)"
        }
    ]
}"""

        user_prompt = f"""Crée une présentation sur: {prompt['topic']}
- Nombre de slides: {prompt['num_slides']}
- Audience: {prompt['target_audience']}
- Langue: {prompt['language']}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )

        training_examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response.choices[0].message.content}
            ]
        })

    return training_examples
```

---

## Préparation du dataset

### Format requis (ChatML)

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Script de conversion

```python
# scripts/prepare_dataset.py

import json
from pathlib import Path

def convert_training_logger_format(input_file: str, output_file: str):
    """Convertit le format TrainingLogger vers ChatML."""

    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            data = json.loads(line)

            # Format ChatML
            chatml = {
                "messages": data["messages"] + [
                    {"role": "assistant", "content": data["response"]}
                ]
            }

            f_out.write(json.dumps(chatml, ensure_ascii=False) + "\n")

def split_dataset(input_file: str, train_ratio: float = 0.9):
    """Split en train/eval."""
    import random

    with open(input_file, "r") as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)

    with open("train.jsonl", "w") as f:
        f.writelines(lines[:split_idx])

    with open("eval.jsonl", "w") as f:
        f.writelines(lines[split_idx:])

    print(f"Train: {split_idx}, Eval: {len(lines) - split_idx}")

def validate_dataset(file_path: str):
    """Valide le format du dataset."""
    errors = []

    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)

                if "messages" not in data:
                    errors.append(f"Line {i}: Missing 'messages' key")
                    continue

                for msg in data["messages"]:
                    if "role" not in msg or "content" not in msg:
                        errors.append(f"Line {i}: Invalid message format")

            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: JSON error - {e}")

    if errors:
        print(f"Found {len(errors)} errors:")
        for err in errors[:10]:
            print(f"  {err}")
    else:
        print("Dataset is valid!")

    return len(errors) == 0

if __name__ == "__main__":
    # 1. Convertir
    convert_training_logger_format(
        "training_dataset.jsonl",
        "chatml_dataset.jsonl"
    )

    # 2. Valider
    validate_dataset("chatml_dataset.jsonl")

    # 3. Split
    split_dataset("chatml_dataset.jsonl")
```

### Taille recommandée du dataset

| Tâche | Exemples min | Exemples recommandés |
|-------|--------------|---------------------|
| Génération de scripts | 500 | 2000+ |
| Génération de code | 1000 | 5000+ |
| Quiz generation | 300 | 1000+ |
| **Total** | **1800** | **8000+** |

---

## Fine-tuning avec Unsloth (Recommandé)

Unsloth est 2x plus rapide et utilise 60% moins de VRAM.

### Installation

```bash
# Créer un environnement
conda create -n unsloth python=3.10
conda activate unsloth

# Installer Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```

### Script d'entraînement

```python
# scripts/train_unsloth.py

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Charger le modèle de base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length=4096,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # QLoRA
)

# 2. Configurer LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# 3. Charger le dataset
dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "eval": "eval.jsonl"
})

# 4. Formater les données
def formatting_prompts_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# 5. Configurer l'entraînement
training_args = TrainingArguments(
    output_dir="./viralify-llama-8b",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=500,  # Ajuster selon la taille du dataset
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    optim="adamw_8bit",
)

# 6. Entraîner
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    dataset_text_field="text",
    max_seq_length=4096,
    args=training_args,
)

trainer.train()

# 7. Sauvegarder
model.save_pretrained("viralify-llama-8b-lora")
tokenizer.save_pretrained("viralify-llama-8b-lora")

# 8. Optionnel: Merger LoRA avec le modèle de base
model.save_pretrained_merged(
    "viralify-llama-8b-merged",
    tokenizer,
    save_method="merged_16bit",
)

# 9. Optionnel: Exporter en GGUF pour Ollama
model.save_pretrained_gguf(
    "viralify-llama-8b-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
```

### Entraînement sur RunPod/Lambda Labs

```bash
# runpod_train.sh

#!/bin/bash
# GPU recommandé: A100 40GB ($1.89/h) ou A100 80GB ($2.49/h)

# Installer les dépendances
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes

# Télécharger le dataset depuis S3/GCS
aws s3 cp s3://viralify-training/train.jsonl .
aws s3 cp s3://viralify-training/eval.jsonl .

# Lancer l'entraînement
python train_unsloth.py

# Upload le modèle
aws s3 cp --recursive viralify-llama-8b-merged s3://viralify-models/
```

---

## Fine-tuning avec Axolotl (Alternative)

Axolotl offre plus de flexibilité mais est plus complexe.

### Configuration YAML

```yaml
# axolotl_config.yaml

base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: true

datasets:
  - path: train.jsonl
    type: sharegpt
    conversation: chatml

dataset_prepared_path: ./prepared_data
val_set_size: 0.05
output_dir: ./viralify-llama-axolotl

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

adapter: qlora
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 3
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 2e-4

train_on_inputs: false
group_by_length: false
bf16: auto
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience: 3
logging_steps: 1
save_steps: 100
eval_steps: 100

warmup_steps: 10
weight_decay: 0.0
```

### Lancer Axolotl

```bash
# Installation
pip install axolotl
pip install flash-attn --no-build-isolation

# Préprocessing
accelerate launch -m axolotl.cli.preprocess axolotl_config.yaml

# Entraînement
accelerate launch -m axolotl.cli.train axolotl_config.yaml

# Inférence test
accelerate launch -m axolotl.cli.inference axolotl_config.yaml \
    --lora_model_dir="./viralify-llama-axolotl"
```

---

## Déploiement du modèle

### Option 1: Ollama (Simple, Local)

```bash
# 1. Créer un Modelfile
cat > Modelfile << 'EOF'
FROM ./viralify-llama-8b-gguf/model-q4_k_m.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"

SYSTEM """Tu es un assistant spécialisé dans la création de contenu éducatif pour Viralify.
Tu génères des scripts de présentation, du code, et des quiz en format JSON."""
EOF

# 2. Créer le modèle Ollama
ollama create viralify-llama -f Modelfile

# 3. Tester
ollama run viralify-llama "Génère un script pour une présentation sur Docker"
```

### Option 2: vLLM (Production, Haute performance)

```bash
# Installation
pip install vllm

# Lancer le serveur
python -m vllm.entrypoints.openai.api_server \
    --model ./viralify-llama-8b-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# Docker
docker run --gpus all -p 8000:8000 \
    -v ./viralify-llama-8b-merged:/model \
    vllm/vllm-openai \
    --model /model \
    --max-model-len 4096
```

### Option 3: RunPod Serverless

```python
# runpod_handler.py

import runpod
from vllm import LLM, SamplingParams

llm = LLM(model="/model", max_model_len=4096)

def handler(event):
    prompt = event["input"]["prompt"]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
    )

    outputs = llm.generate([prompt], sampling_params)

    return {"output": outputs[0].outputs[0].text}

runpod.serverless.start({"handler": handler})
```

---

## Intégration avec Viralify

### 1. Ajouter le provider custom

```python
# services/shared/llm_provider.py

class ViralifyLlamaProvider(LLMProvider):
    """Provider pour le modèle fine-tuné Viralify."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="not-needed"  # vLLM n'a pas besoin de clé
        )
        self.model = "viralify-llama-8b"

    def generate(self, messages: List[Dict], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 2048),
        )
        return response.choices[0].message.content
```

### 2. Configurer dans docker-compose

```yaml
# docker-compose.prod.yml

services:
  viralify-llm:
    image: vllm/vllm-openai:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models/viralify-llama-8b:/model
    command: >
      --model /model
      --max-model-len 4096
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    networks:
      - viralify-network

  nexus-engine:
    environment:
      - LLM_PROVIDER=viralify
      - VIRALIFY_LLM_URL=http://viralify-llm:8000
```

### 3. Variables d'environnement

```bash
# .env
LLM_PROVIDER=viralify
VIRALIFY_LLM_URL=http://viralify-llm:8000
VIRALIFY_MODEL_NAME=viralify-llama-8b
```

---

## Estimation des coûts

### Coût d'entraînement (one-time)

| Ressource | Durée | Coût |
|-----------|-------|------|
| A100 40GB (RunPod) | ~2h pour 5000 exemples | ~$4 |
| A100 80GB (Lambda) | ~1.5h pour 5000 exemples | ~$5 |
| Génération données (GPT-4) | 5000 exemples | ~$25 |
| **Total** | | **~$30-35** |

### Coût d'inférence (mensuel)

| Solution | Coût/mois | Tokens/mois |
|----------|-----------|-------------|
| GPT-4o (actuel) | ~$500 | 50M |
| Groq | ~$35 | 50M |
| **Self-hosted A100** | ~$200 | Illimité |
| **RunPod Serverless** | ~$50 | 50M |

### ROI

- Investissement initial: ~$35
- Économie mensuelle: $300-450
- **Retour sur investissement: < 1 semaine**

---

## Checklist de déploiement

- [ ] Générer 5000+ exemples synthétiques
- [ ] Valider le format du dataset
- [ ] Entraîner avec Unsloth (2h sur A100)
- [ ] Exporter en GGUF pour test local
- [ ] Tester avec Ollama localement
- [ ] Déployer avec vLLM sur GPU dédié
- [ ] Ajouter le provider Viralify
- [ ] Tester en staging
- [ ] Déployer en production
- [ ] Monitorer la qualité des outputs

---

## Ressources

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Axolotl GitHub](https://github.com/OpenAccess-AI-Collective/axolotl)
- [vLLM Documentation](https://docs.vllm.ai/)
- [RunPod Serverless](https://docs.runpod.io/serverless)
- [Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
