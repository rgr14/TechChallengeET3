from main import main
import streamlit as st
import json
import sys
import os

# Adicionar o diretório atual ao path para importar o main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import main
    import_success = True
except ImportError as e:
    import_success = False
    error_message = str(e)

# Configuração da página
st.set_page_config(
    page_title="Formulário de Dados do Imóvel",
    layout="wide"
)

st.title("Formulário de Dados do Imóvel")

# Verificar se o import funcionou
if not import_success:
    st.error(f"Erro ao importar a função main: {error_message}")
    st.info("Certifique-se que o arquivo 'main.py' está no mesmo diretório deste script.")
    st.stop()
else:
    st.success("Função main importada com sucesso!")

st.markdown("---")

# Criação do formulário
with st.form("dados_imovel"):
    st.subheader("Informações do Imóvel")
    
    # Criando colunas para melhor organização
    col1, col2 = st.columns(2)
    
    with col1:
        # Campos booleanos como checkboxes
        pet = st.checkbox("Aceita Pet", value=False)
        mobiliado = st.checkbox("Mobiliado", value=False)
        metro_proximo = st.checkbox("Próximo ao Metrô", value=False)
        
        # Campos numéricos - características do imóvel
        area = st.number_input("Área (m²)", min_value=0.1, value=50.0, step=1.0)
        quartos = st.number_input("Número de Quartos", min_value=0, value=2, step=1)
        suite = st.number_input("Número de Suítes", min_value=0, value=1, step=1)
        banheiros = st.number_input("Número de Banheiros", min_value=0, value=1, step=1)
    
    with col2:
        # Campos monetários
        st.subheader("Valores Financeiros")
        aluguel = st.number_input("Aluguel (R$)", min_value=0.0, value=4000.0, step=50.0)
        condominio = st.number_input("Condomínio (R$)", min_value=0.0, value=800.0, step=50.0)
        iptu = st.number_input("IPTU (R$)", min_value=0.0, value=150.0, step=10.0)
        seguro_incendio = st.number_input("Seguro Incêndio (R$)", min_value=0.0, value=30.0, step=5.0)
        taxa_servico = st.number_input("Taxa de Serviço (R$)", min_value=0.0, value=0.0, step=10.0)
        total = st.number_input("Total (R$)", min_value=0.0, value=7000.0, step=100.0)
        
    
    # Botão de envio
    submitted = st.form_submit_button("Enviar Dados", type="primary")
    
    if submitted:
        try:
            # Verificar se área é válida para o cálculo
            if area <= 0:
                st.error("A área deve ser maior que zero para calcular o custo por m²")
                st.stop()
            
            # Calcular custo por m²
            cost_per_m2 = float(aluguel) / float(area)
            
            # Criação do dicionário com os dados
            filtro = {
                "pet": 1 if pet else 0,
                "mobiliado": 1 if mobiliado else 0,
                "metro_proximo": 1 if metro_proximo else 0,
                "area": float(area),  # Sempre float
                "quartos": int(quartos),  # Inteiros permanecem como int
                "suite": int(suite),  # Inteiros permanecem como int
                "banheiros": int(banheiros),  # Inteiros permanecem como int
                "aluguel": float(aluguel),  # Sempre float
                "condominio": float(condominio),  # Sempre float
                "iptu": float(iptu),  # Sempre float
                "seguro_incendio": float(seguro_incendio),  # Sempre float
                "taxa_servico": float(taxa_servico),  # Sempre float
                "total": float(total),  # Sempre float
                "cost_per_m2": round(cost_per_m2, 2) #Calculado automaticamente
            }
            
            # Salvar no arquivo JSON
            try:
                config_path = os.path.join("config", "user_input.json")
                
                # Criar pasta config se não existir
                os.makedirs("config", exist_ok=True)
                
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(filtro, f, indent=4, ensure_ascii=False)
                
                st.success(f"Dados salvos em {config_path}")
                
            except Exception as e:
                st.error(f"Erro ao salvar arquivo JSON: {str(e)}")
            
            # Mostrar os dados que serão enviados
            st.subheader("Dados processados:")
            st.json(filtro)
            
            # Chamada da função main com os dados
            with st.spinner("Executando modelo..."):
                main(filtro)
                
        except Exception as e:
            st.error(f"Erro ao processar os dados: {str(e)}")
            st.error("Verifique se a função main está funcionando corretamente.")
            
            # Mostrar traceback para debugging
            import traceback
            st.code(traceback.format_exc())