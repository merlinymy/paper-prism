#!/bin/bash
# Download 20 landmark LLM and RAG papers from arXiv
# These are all publicly available research papers

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIR="${SCRIPT_DIR}/demo_papers"
mkdir -p "$DIR"
cd "$DIR"

echo "Downloading 20 LLM/RAG papers to $DIR..."
echo ""

# Function to download with progress
download() {
    local url="$1"
    local filename="$2"
    if [ -f "$filename" ]; then
        echo "  SKIP (exists): $filename"
    else
        echo "  Downloading: $filename"
        curl -sL -o "$filename" "$url"
        if [ $? -eq 0 ] && [ -s "$filename" ]; then
            echo "  OK: $(du -h "$filename" | cut -f1)"
        else
            echo "  FAILED: $filename"
            rm -f "$filename"
        fi
    fi
}

# 1. Attention Is All You Need (Vaswani et al., 2017)
download "https://arxiv.org/pdf/1706.03762" "Vaswani_2017_Attention_Is_All_You_Need.pdf"

# 2. BERT (Devlin et al., 2019)
download "https://arxiv.org/pdf/1810.04805" "Devlin_2019_BERT_Pretraining.pdf"

# 3. GPT-2 (Radford et al., 2019)
download "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" "Radford_2019_GPT2_Language_Models.pdf"

# 4. GPT-3 / Language Models are Few-Shot Learners (Brown et al., 2020)
download "https://arxiv.org/pdf/2005.14165" "Brown_2020_GPT3_Few_Shot_Learners.pdf"

# 5. RAG - Retrieval-Augmented Generation (Lewis et al., 2020)
download "https://arxiv.org/pdf/2005.11401" "Lewis_2020_Retrieval_Augmented_Generation.pdf"

# 6. Dense Passage Retrieval (Karpukhin et al., 2020)
download "https://arxiv.org/pdf/2004.04906" "Karpukhin_2020_Dense_Passage_Retrieval.pdf"

# 7. ColBERT (Khattab & Zaharia, 2020)
download "https://arxiv.org/pdf/2004.12832" "Khattab_2020_ColBERT_Passage_Search.pdf"

# 8. InstructGPT / Training with Human Feedback (Ouyang et al., 2022)
download "https://arxiv.org/pdf/2203.02155" "Ouyang_2022_InstructGPT_RLHF.pdf"

# 9. Chain-of-Thought Prompting (Wei et al., 2022)
download "https://arxiv.org/pdf/2201.11903" "Wei_2022_Chain_of_Thought_Prompting.pdf"

# 10. Constitutional AI (Bai et al., 2022)
download "https://arxiv.org/pdf/2212.08073" "Bai_2022_Constitutional_AI.pdf"

# 11. HyDE - Hypothetical Document Embeddings (Gao et al., 2022)
download "https://arxiv.org/pdf/2212.10496" "Gao_2022_HyDE_Zero_Shot_Dense_Retrieval.pdf"

# 12. LLaMA (Touvron et al., 2023)
download "https://arxiv.org/pdf/2302.13971" "Touvron_2023_LLaMA_Open_Foundation_Models.pdf"

# 13. Self-RAG (Asai et al., 2023)
download "https://arxiv.org/pdf/2310.11511" "Asai_2023_Self_RAG.pdf"

# 14. Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2024)
download "https://arxiv.org/pdf/2312.10997" "Gao_2024_RAG_Survey.pdf"

# 15. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024)
download "https://arxiv.org/pdf/2401.18059" "Sarthi_2024_RAPTOR_Tree_Retrieval.pdf"

# 16. Mistral 7B (Jiang et al., 2023)
download "https://arxiv.org/pdf/2310.06825" "Jiang_2023_Mistral_7B.pdf"

# 17. LoRA: Low-Rank Adaptation (Hu et al., 2021)
download "https://arxiv.org/pdf/2106.09685" "Hu_2021_LoRA_Low_Rank_Adaptation.pdf"

# 18. Tree of Thoughts (Yao et al., 2023)
download "https://arxiv.org/pdf/2305.10601" "Yao_2023_Tree_of_Thoughts.pdf"

# 19. Toolformer (Schick et al., 2023)
download "https://arxiv.org/pdf/2302.04761" "Schick_2023_Toolformer.pdf"

# 20. Lost in the Middle (Liu et al., 2023)
download "https://arxiv.org/pdf/2307.03172" "Liu_2023_Lost_in_the_Middle.pdf"

echo ""
echo "Download complete. Papers in $DIR:"
ls -la "$DIR"/*.pdf 2>/dev/null | wc -l
echo "Total size: $(du -sh "$DIR" | cut -f1)"
