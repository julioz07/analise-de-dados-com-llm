# 📊 Análise de Dados com LLM - Dashboard Inteligente para Restaurante & Hotel

> **Sistema integrado de análise de dados, Machine Learning e LLM para otimização de negócios no setor de hospitalidade**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

## 🚀 Demo Online

👉 **[🌐 Acesse o Dashboard Aqui](https://dashboarhotelerestaurentecomllm0jtjn.streamlit.app/)**

---

## � Sobre o Projeto

Este projeto representa uma **avaliação acadêmica completa** que integra **Análise de Dados**, **Machine Learning** e **Large Language Models (LLM)** para criar um sistema inteligente de apoio à decisão para negócios de hospitalidade.

### 🎯 **Objetivos**
- Demonstrar competências em **análise exploratória de dados**
- Implementar **modelos de Machine Learning** para previsões precisas
- Integrar **LLMs** para insights automatizados
- Criar **dashboard interativo** para visualização e tomada de decisão
- Aplicar **boas práticas** de desenvolvimento e deploy

---

## �📈 Funcionalidades

### 🏠 **Visão Geral**
- 📊 **Métricas Executivas**: KPIs principais dos dois negócios
- � **Comparação de Performance**: Receitas, clientes e tendências
- 🎯 **Análise Cruzada**: Insights entre restaurante e hotel
- 💰 **Projeções Financeiras**: Estimativas baseadas em dados históricos

### �📊 **Análise Exploratória Interativa**
- 🔍 **Exploração por Dataset**: Análise detalhada separada (Restaurante/Hotel)
- 🔗 **Matriz de Correlação**: Identificação de padrões e relacionamentos
- 📅 **Análise Temporal**: Tendências e sazonalidades (quando disponível)
- 🤖 **Insights Automáticos**: Descobertas geradas por IA

### 🤖 **Machine Learning Avançado**
- 🎯 **Modelos de Regressão**: Previsão de gastos e receitas
- 🔮 **Modelos de Classificação**: Probabilidade de retorno e uso de serviços
- 📊 **Visualizações ML**: Gráficos interativos de performance
- ⚡ **Comparação de Modelos**: RandomForest, SVM, Linear Models

### 🔮 **Sistema de Previsões Inteligente**
- 🍽️ **Restaurante**: Previsão de gasto total e probabilidade de retorno
- 🏨 **Hotel**: Previsão de gasto e probabilidade de uso do SPA
- 📱 **Interface Intuitiva**: Inputs dinâmicos para cenários personalizados
- 📈 **Resultados em Tempo Real**: Previsões instantâneas

### 💡 **Insights LLM Integrados**
- 🧠 **Análises Automatizadas**: Interpretação inteligente dos dados
- 💼 **Estratégias Comerciais**: Recomendações baseadas em IA
- 📝 **Relatórios Narrativos**: Explicações detalhadas dos padrões encontrados
- 🎯 **Recomendações Personalizadas**: Sugestões específicas por segmento

---

## 🛠️ Stack Tecnológica

### **Frontend & Visualização**
- **[Streamlit](https://streamlit.io/)**: Framework web interativo
- **[Plotly](https://plotly.com/)**: Gráficos interativos avançados
- **[Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)**: Visualizações estáticas

### **Análise de Dados**
- **[Pandas](https://pandas.pydata.org/)**: Manipulação e análise de dados
- **[NumPy](https://numpy.org/)**: Computação numérica
- **[SciPy](https://scipy.org/)**: Análises estatísticas avançadas

### **Machine Learning**
- **[Scikit-learn](https://scikit-learn.org/)**: Modelos de ML e métricas
- **RandomForestRegressor & RandomForestClassifier**: Modelos ensemble
- **SVM (SVR & SVC)**: Support Vector Machines
- **Linear & Logistic Regression**: Modelos lineares

### **Deploy & Infraestrutura**
- **[Streamlit Cloud](https://streamlit.io/cloud)**: Deploy automático
- **[GitHub](https://github.com/)**: Controle de versão
- **Requirements Management**: Dependências otimizadas

---

## 📊 Estrutura dos Dados

### 🍽️ **Dataset Restaurante**
```
📁 Datasets_clean/restaurante_clean.csv
📁 Datasets_ML/restaurante_ml.csv
```
- **Clientes**: Perfil demográfico e comportamental
- **Gastos**: Valores detalhados por categoria
- **Retorno**: Histórico de visitas e fidelização

### 🏨 **Dataset Hotel**
```
📁 Datasets_clean/hotel_clean.csv  
📁 Datasets_ML/hotel_ml.csv
```
- **Reservas**: Dados de ocupação e estadia
- **Serviços**: Utilização de SPA e amenities
- **Gastos**: Breakdown detalhado por serviço

### 👥 **Dataset Clientes**
```
📁 Datasets_clean/clientes.csv
```
- **Demografia**: Idade, localização, perfil
- **Comportamento**: Padrões de consumo cross-business

---

## 🎯 Resultados Machine Learning

### 🍽️ **Restaurante - Performance Excepcional**
| Modelo | Tipo | Métrica | Score |
|--------|------|---------|-------|
| RandomForest | Regressão | R² Score | **1.00** ✨ |
| RandomForest | Classificação | Accuracy | **100%** ✨ |
| SVM | Regressão | MAE | **0.00** |
| Linear | Classificação | Precision | **100%** |

### 🏨 **Hotel - Modelos Otimizados**
| Modelo | Tipo | Aplicação | Performance |
|--------|------|-----------|-------------|
| RandomForest | Regressão | Previsão Gastos | **Alta** 📈 |
| SVM | Classificação | Uso SPA | **Otimizada** 🎯 |
| Ensemble | Híbrido | Cenários Complexos | **Robusta** 💪 |

---

## 🚀 Como Executar Localmente

### **Pré-requisitos**
```bash
Python 3.8+
Git
```

### **Instalação**
```bash
# 1. Clonar repositório
git clone https://github.com/julioz07/analise-de-dados-com-llm.git
cd analise-de-dados-com-llm

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Executar aplicação
streamlit run dashboard_streamlit.py
```

### **Acesso Local**
```
🌐 http://localhost:8501
```

---

## 📱 Screenshots & Demo

### 🏠 **Dashboard Principal**
![Dashboard Overview](https://via.placeholder.com/800x400/FF6B6B/FFFFFF?text=Dashboard+Overview)

### 📊 **Análise Exploratória**
![Análise Exploratória](https://via.placeholder.com/800x400/4ECDC4/FFFFFF?text=Análise+Exploratória)

### 🤖 **Machine Learning**
![ML Results](https://via.placeholder.com/800x400/45B7D1/FFFFFF?text=Machine+Learning+Results)

### 🔮 **Sistema de Previsões**
![Previsões](https://via.placeholder.com/800x400/96CEB4/FFFFFF?text=Sistema+de+Previsões)

---

## 👥 Equipe de Desenvolvimento

Este projeto foi desenvolvido como trabalho acadêmico pela equipe:

### 🧑‍💻 **Julio** 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/julio-oliveira-dev/)
- **Role**: Tech Lead & Full-Stack Development
- **Contribuições**: Arquitetura, ML Pipeline, Dashboard Frontend

### 👩‍💻 **Joana**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](#)
- **Role**: Data Scientist & Analytics
- **Contribuições**: Análise Exploratória, Feature Engineering

### 👨‍💻 **Nuno**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](#)
- **Role**: Machine Learning Engineer
- **Contribuições**: Modelos ML, Otimização de Performance

### 👩‍💻 **Tânia**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](#)
- **Role**: Data Analyst & Visualization
- **Contribuições**: Visualizações, UX/UI, Insights

---

## 📂 Estrutura do Projeto

```
📦 analise-de-dados-com-llm/
├── 📄 dashboard_streamlit.py       # 🚀 Aplicação principal Streamlit
├── 📄 config.py                    # ⚙️ Configurações do projeto
├── 📄 requirements.txt             # 📦 Dependências Python
├── 📁 .streamlit/                  # 🎨 Configurações Streamlit
│   └── 📄 config.toml              #     Tema e configurações UI
├── 📁 Datasets_clean/              # 🧹 Dados limpos e processados
│   ├── 📄 restaurante_clean.csv    #     Dataset restaurante
│   ├── 📄 hotel_clean.csv          #     Dataset hotel
│   └── 📄 clientes.csv             #     Dataset clientes
├── 📁 Datasets_ML/                 # 🤖 Dados preparados para ML
│   ├── 📄 restaurante_ml.csv       #     Features restaurante
│   └── 📄 hotel_ml.csv             #     Features hotel
├── 📄 .gitignore                   # 🚫 Arquivos ignorados
├── 📄 README.md                    # 📖 Documentação principal
└── 📄 DEPLOY_STREAMLIT_CLOUD.md    # 🚀 Guia de deploy
```

---

## 🎓 Contexto Acadêmico

### **Disciplina**: Análise de Dados com Large Language Models
### **Instituição**: [Nome da Instituição]
### **Período**: 2025.1
### **Tipo**: Avaliação Prática Integrada

### **Competências Avaliadas**
- ✅ **Manipulação e Análise de Dados** com Pandas/NumPy
- ✅ **Visualização Interativa** com Plotly/Streamlit  
- ✅ **Machine Learning** com Scikit-learn
- ✅ **Integração de LLMs** para insights automatizados
- ✅ **Deploy e Produção** com Streamlit Cloud
- ✅ **Trabalho em Equipe** e versionamento Git

---

## 🔄 Próximas Melhorias

### **Versão 2.0 (Roadmap)**
- 🔐 **Autenticação**: Sistema de login para diferentes perfis
- 📊 **Dashboard Avançado**: Métricas em tempo real
- 🤖 **LLM Avançado**: Integração com GPT-4/Claude para insights mais sofisticados
- 📱 **Mobile App**: Versão nativa para smartphones
- 🔗 **API REST**: Endpoints para integração externa
- 📈 **A/B Testing**: Framework para testes de estratégias

### **Melhorias Técnicas**
- ⚡ **Performance**: Cache inteligente e otimização de queries
- 🛡️ **Segurança**: Criptografia de dados sensíveis
- 📊 **Monitoring**: Logs e métricas de uso
- 🔄 **CI/CD**: Pipeline automatizado de deploy

---

## 📄 Licença

Este projeto foi desenvolvido para fins **educacionais** como parte de uma avaliação acadêmica.

### **Uso Permitido**
- ✅ Estudo e aprendizado
- ✅ Referência para projetos similares
- ✅ Demonstração de competências técnicas

### **Restrições**
- ❌ Uso comercial sem autorização
- ❌ Cópia integral sem atribuição
- ❌ Redistribuição dos datasets sem permissão

---

## 🤝 Contribuições

Embora este seja um projeto acadêmico, feedbacks e sugestões são sempre bem-vindos!

### **Como Contribuir**
1. 🍴 Fork o projeto
2. 🌿 Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push para a branch (`git push origin feature/AmazingFeature`)
5. 🔄 Abra um Pull Request

---

## 📞 Contato

Para dúvidas, sugestões ou oportunidades de colaboração:

**📧 Email da Equipe**: [contato@projeto-analise-llm.com]  
**💼 LinkedIn**: Conecte-se com qualquer membro da equipe  
**🐙 GitHub**: [@julioz07](https://github.com/julioz07)

---

<div align="center">

### 🌟 **Se este projeto foi útil, deixe uma ⭐ no repositório!**

**Desenvolvido com ❤️ pela Equipe Julio, Joana, Nuno & Tânia**

![Footer](https://via.placeholder.com/800x100/FF6B6B/FFFFFF?text=Análise+de+Dados+com+LLM+-+2025)

</div>
