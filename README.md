# Text-to-SQL Generation with PEFT: Bridging Natural Language and Databases

A comprehensive implementation of Text-to-SQL generation using Parameter-Efficient Fine-Tuning (PEFT) techniques, systematically exploring baseline models, prompt engineering, and advanced fine-tuning methods.

## Project Overview

This project addresses the challenge of converting natural language queries into accurate SQL statements, making database access more intuitive for non-technical users. The implementation follows a structured, research-driven approach with rigorous benchmarking at each stage.

### Key Features
- **Baseline Model Evaluation**: Comprehensive testing of SQLCoder and CodeLlama models
- **Systematic Prompt Engineering**: Multiple prompt strategies with quantitative analysis
- **Advanced Fine-tuning**: Implementation of LoRA and qLoRA techniques
- **Robust Evaluation**: Multiple metrics including Exact Match, BLEU, and execution accuracy
- **User-friendly Deployment**: Gradio-based web interface for practical usage

## Dataset

The project utilizes the [**Spider Dataset**](https://huggingface.co/datasets/xlangai/spider), the gold standard for Text-to-SQL research, featuring:
- Large-scale complex and cross-domain semantic parsing
- Diverse natural language queries with corresponding SQL statements
- Multiple database schemas for comprehensive evaluation
- Human-annotated by Yale students for quality assurance

## Project Architecture

### Stage 1: Environment Setup & Problem Framing
- Library installation and configuration
- Hugging Face API setup
- Problem definition and use case articulation

### Stage 2: Dataset Exploration
- Spider dataset loading and analysis
- Statistical visualization of dataset properties
- Database distribution analysis and complexity assessment

### Stage 3: Baseline Implementation
- **SQLCoder** (`defog/sqlcoder-7b-2`): Specialized Text-to-SQL model
- **CodeLlama** (`codellama/CodeLlama-7b-Instruct-hf`): General-purpose code generation model
- Quantitative baseline performance evaluation

### Stage 4: Prompt Engineering
- **Basic Prompt**: Simple, direct query format
- **Instruction Prompt**: Structured question-answer format
- **Few-shot Prompt**: Context-aware with examples
- Systematic evaluation and optimization

### Stage 5: Parameter-Efficient Fine-tuning (PEFT)
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning with minimal parameters
- **qLoRA (Quantized LoRA)**: Memory-efficient quantized fine-tuning
- **Alternative PEFT Methods**: Prompt-tuning and P-tuning v2 exploration

### Stage 6: Deployment & User Interface
- Gradio-based web application
- Real-time SQL generation interface
- User-friendly demonstration platform

##  Evaluation Metrics

### Primary Metrics
1. **Exact Match (EM)**: Measures perfect SQL query matches
2. **BLEU Score**: Evaluates textual similarity between generated and reference queries
3. **Execution-based Accuracy**: Validates practical query correctness through database execution

### Example Evaluation
```python
# Natural Language Query
"List all customers who spent more than $5000 in 2024"

# Generated SQL Query
SELECT customer_name, SUM(amount_spent) as total_spent
FROM transactions
WHERE YEAR(transaction_date) = 2024
GROUP BY customer_name
HAVING total_spent > 5000;
```

##  Technical Implementation

### Core Technologies
- **Transformers**: Hugging Face transformers library for model handling
- **PEFT**: Parameter-efficient fine-tuning implementation
- **BitsAndBytesConfig**: 4-bit quantization for memory efficiency
- **Gradio**: Web interface deployment
- **Datasets**: Spider dataset integration

### Model Configurations
```python
# LoRA Configuration
lora_config = LoraConfig(
    r=16,                    # Low-rank dimension
    lora_alpha=32,          # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Quantization Configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
```

##  Getting Started

### Prerequisites
```bash
pip install transformers datasets accelerate peft gradio pandas matplotlib seaborn huggingface_hub fsspec bitsandbytes
```

### Quick Start
1. **Clone the repository**
2. **Set up Hugging Face authentication**
3. **Run the notebook sections sequentially**
4. **Deploy the Gradio interface**

### Usage Example
```python
# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained("path/to/finetuned_model")

# Generate SQL from natural language
def generate_sql(question):
    prompt = f"Generate SQL query: {question}"
    response = pipeline(prompt, max_length=200)[0]['generated_text']
    return response

# Example usage
query = "Show total sales by region for Q1 2024"
sql_result = generate_sql(query)
```

## Performance Analysis

### Baseline Comparison
| Model | Exact Match (%) | BLEU Score | Execution Accuracy (%) |
|-------|----------------|------------|------------------------|
| SQLCoder | TBD | TBD | TBD |
| CodeLlama | TBD | TBD | TBD |

### Fine-tuning Results
| Method | Performance Improvement | Memory Usage | Training Time |
|--------|------------------------|--------------|---------------|
| LoRA | TBD | Low | Fast |
| qLoRA | TBD | Ultra-low | Moderate |

##  Research Papers & Resources

### Core Papers
1. **Spider Dataset**: Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., ... & Radev, D. (2018). [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887). *EMNLP 2018*.

2. **LoRA**: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). *ICLR 2022*.

3. **qLoRA**: Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314). *NeurIPS 2023*.

4. **P-tuning v2**: Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z., & Tang, J. (2021). [P-tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602). *ACL 2022*.

### Additional Resources
- [SQLCoder GitHub Repository](https://github.com/defog-ai/sqlcoder)
- [CodeLlama Model Card](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
- [Spider Dataset Card](https://huggingface.co/datasets/xlangai/spider)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)

##  Prompt Engineering Strategies

### Implemented Strategies
1. **Basic Prompt**: Direct query generation
   ```
   "Write SQL query for: {question}"
   ```

2. **Instruction Prompt**: Structured format
   ```
   "Given the following question, produce the corresponding SQL query:
   Question: {question}
   SQL query:"
   ```

3. **Few-shot Prompt**: Context-aware examples
   ```
   "Question: How many employees joined after January 2020?
   SQL: SELECT COUNT(*) FROM employees WHERE join_date > '2020-01-31';
   
   Question: {question}
   SQL:"
   ```

##  Advanced Features

### Memory Optimization
- 4-bit quantization for large model deployment
- CPU offloading for memory-constrained environments
- Gradient accumulation for effective batch processing

### Scalability Considerations
- Modular architecture for easy extension
- Support for multiple database backends
- Configurable model parameters for different use cases

##  Deployment

### Gradio Interface
```python
import gradio as gr

def generate_sql(question):
    prompt = f"Generate SQL query: {question}"
    response = pipe(prompt, max_length=200)[0]['generated_text']
    return response

interface = gr.Interface(
    fn=generate_sql,
    inputs="text",
    outputs="text",
    title="Text to SQL Generator",
    description="Convert natural language questions to SQL queries."
)

interface.launch()
```

##  Future Enhancements

### Potential Extensions
- **SQL Validation Integration**: Automated correctness validation
- **Cross-Domain Generalization**: Multi-database schema support
- **Explainability Analysis**: Attention visualization for model interpretation
- **RAG Integration**: Retrieval-augmented generation for complex queries
- **Chain-of-Thought Prompting**: Step-by-step reasoning for complex queries

##  Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Spider dataset creators at Yale University
- Hugging Face team for the transformers and PEFT libraries
- Open-source contributors to SQLCoder and CodeLlama projects
- Research community for advancing Text-to-SQL generation

---

**Note**: This is a research and educational project. Performance metrics and specific results may vary based on hardware configurations and dataset variations. Always validate generated SQL queries before production use. 