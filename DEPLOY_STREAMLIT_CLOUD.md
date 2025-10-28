# 🚀 DEPLOY NO STREAMLIT CLOUD - PASSO A PASSO

## 📋 Pré-requisitos
- ✅ Conta no GitHub
- ✅ Conta no Streamlit Cloud
- ✅ Arquivos desta pasta

## 🎯 PASSO 1: Criar Repositório GitHub

### Via GitHub Web:
1. Acesse: https://github.com
2. Clique em "New repository"
3. Nome: `dashboard-restaurante-hotel`
4. Deixe como **público**
5. ✅ Initialize with README
6. Clique "Create repository"

### Via Git Local:
```bash
# Na pasta streamlit_cloud_deploy
git init
git add .
git commit -m "Initial commit - Dashboard Streamlit"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/dashboard-restaurante-hotel.git
git push -u origin main
```

## 🎯 PASSO 2: Deploy no Streamlit Cloud

1. **Acesse:** https://share.streamlit.io
2. **Clique:** "Deploy a public app from GitHub"
3. **Preencha:**
   - Repository: `SEU_USUARIO/dashboard-restaurante-hotel`
   - Branch: `main`
   - Main file path: `dashboard_streamlit.py`
4. **Clique:** "Deploy!"

## ⏱️ PASSO 3: Aguardar Deploy

- **Tempo estimado:** 2-5 minutos
- **Status:** Acompanhe na interface
- **URL final:** `https://seu-app.streamlit.app`

## 🔧 PASSO 4: Configurações (Opcional)

### Personalizar URL:
- Vá em "Settings" do app
- Altere o nome se desejar

### Variáveis de Ambiente:
- Se precisar de API keys
- Configure em "Secrets"

## 🐛 SOLUÇÃO DE PROBLEMAS

### Erro de Dependencies:
- Verificar `requirements.txt`
- Testar localmente primeiro

### Erro de Dados:
- Verificar se datasets estão no repo
- Caminhos relativos corretos

### App não carrega:
- Verificar logs no Streamlit Cloud
- Verificar nome do arquivo principal

## 🔄 ATUALIZAÇÕES

Para atualizar o app:
1. Faça mudanças no código
2. Commit e push para GitHub
3. O Streamlit Cloud atualiza automaticamente!

## 📱 COMPARTILHAMENTO

Após o deploy:
- ✅ URL pública funcional
- ✅ Acessível de qualquer dispositivo
- ✅ Sem necessidade de instalação
- ✅ Sempre na versão mais recente

## 🎉 RESULTADO FINAL

Seu dashboard estará disponível em:
`https://[nome-do-app].streamlit.app`

**Exemplo:** `https://dashboard-restaurante-hotel.streamlit.app`
