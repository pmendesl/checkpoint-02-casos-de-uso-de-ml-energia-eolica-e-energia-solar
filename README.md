# Soluções em Energias Renováveis e Sustentáveis

Integrates:
Pedro Mendes RM: 562242
Leonardo RM: 565564
Alexandre RM: 563346
Guilherme Peres RM: 563981
Gabriel de Matos RM: 565218

Este repositório contém a resolução dos exercícios da disciplina "Soluções em Energias Renováveis e Sustentáveis". O projeto aborda tarefas de **Classificação** e **Regressão** aplicadas a datasets de energia solar e eólica, utilizando Python (com as bibliotecas Pandas e Scikit-learn) e Orange Data Mining.

## 📝 Instruções da Entrega

*   **Atividade:** Pode ser desenvolvida em grupo.
*   **Submissão:** Apenas um integrante do grupo deve submeter o link do repositório.
*   **Formato:** Não enviar arquivos `.txt` ou `.pdf`, apenas o link do repositório do GitHub.

## 📊 Datasets Utilizados

### 1. Solar Radiation Prediction Dataset
- **Link para Download:** [Kaggle - Solar Energy](https://www.kaggle.com/datasets/dronio/SolarEnergy )
- **Descrição:** Este dataset contém dados meteorológicos de quatro estações de medição em Hawaii. As variáveis incluem temperatura, pressão, umidade, direção e velocidade do vento, além da radiação solar (em W/m²), que é a nossa variável de interesse para a tarefa de classificação.

### 2. Wind Turbine Scada Dataset
- **Link para Download:** [Kaggle - Wind Turbine Scada](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset )
- **Descrição:** Este dataset contém dados de um sistema SCADA (Supervisory Control and Data Acquisition) de uma turbina eólica na Turquia. Ele registra a potência ativa gerada (kW), velocidade e direção do vento, e a curva de potência teórica a cada 10 minutos. É utilizado para a tarefa de regressão.

## 🐍 Resolução em Python

### Pré-requisitos

Antes de executar os scripts, certifique-se de ter as bibliotecas necessárias instaladas.

```bash
pip install pandas scikit-learn numpy matplotlib seaborn
```

### Exercício 1 – Classificação (Solar)

**Objetivo:** Classificar os períodos em "Alta Radiação" e "Baixa Radiação" com base em dados meteorológicos.

**Modelos Utilizados:** Árvore de Decisão, Random Forest, SVM.
**Métricas de Avaliação:** Acurácia e Matriz de Confusão.

```python
# 1. Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Carregamento e Preparação dos Dados ---
# Certifique-se de que o arquivo 'SolarPrediction.csv' está na mesma pasta.
try:
    df = pd.read_csv('SolarPrediction.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'SolarPrediction.csv' não encontrado.")
    exit()

# --- Pré-processamento ---
# Criar a variável-alvo categórica usando a mediana como limiar
radiation_median = df['Radiation'].median()
df['Radiation_Level'] = np.where(df['Radiation'] <= radiation_median, 0, 1) # 0: Baixa, 1: Alta

# Selecionar features e target
features = ['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']
target = 'Radiation_Level'
X = df[features]
y = df[target]

# Dividir dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizar os atributos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Treinamento e Avaliação dos Modelos ---
models = {
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {"accuracy": accuracy, "confusion_matrix": cm}
    print(f"--- {name} ---\nAcurácia: {accuracy:.4f}\nMatriz de Confusão:\n{cm}\n")

# --- Comparação e Visualização ---
plt.figure(figsize=(18, 5))
for i, (name, result) in enumerate(results.items()):
    plt.subplot(1, 3, i + 1)
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Baixa', 'Alta'], yticklabels=['Baixa', 'Alta'])
    plt.title(f'Matriz de Confusão - {name}\nAcurácia: {result["accuracy"]:.4f}')
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
plt.tight_layout()
plt.show()
```

### Exercício 2 – Regressão (Eólica)

**Objetivo:** Prever a potência gerada (kW) por uma turbina eólica com base na velocidade e direção do vento.

**Modelos Utilizados:** Regressão Linear, Regressão de Árvores, Random Forest Regressor.
**Métricas de Avaliação:** RMSE e R².

```python
# 1. Importação das bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Carregamento e Preparação dos Dados ---
# Certifique-se de que o arquivo 'T1.csv' está na mesma pasta.
try:
    df = pd.read_csv('T1.csv')
except FileNotFoundError:
    print("Erro: Arquivo 'T1.csv' não encontrado.")
    exit()

# Renomear colunas e filtrar dados inválidos
df.columns = ['DateTime', 'ActivePower_kW', 'WindSpeed_ms', 'Theoretical_Power_Curve_KWH', 'WindDirection_Degrees']
df_filtered = df[(df['ActivePower_kW'] >= 0) & (df['WindSpeed_ms'] > 0)].copy()

# --- Pré-processamento ---
# Selecionar features e target
features = ['WindSpeed_ms', 'WindDirection_Degrees']
target = 'ActivePower_kW'
X = df_filtered[features]
y = df_filtered[target]

# Dividir dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os atributos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Treinamento e Avaliação dos Modelos ---
models = {
    "Regressão Linear": LinearRegression(),
    "Árvore de Regressão": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {"RMSE": rmse, "R2": r2, "predictions": y_pred}
    print(f"--- {name} ---\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\n")

# --- Comparação e Visualização ---
plt.figure(figsize=(18, 5))
for i, (name, result) in enumerate(results.items()):
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(x=y_test, y=result['predictions'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title(f'{name}\nRMSE: {result["RMSE"]:.2f} | R²: {result["R2"]:.2f}')
    plt.xlabel('Potência Real (kW)')
    plt.ylabel('Potência Prevista (kW)')
    plt.grid(True)
plt.tight_layout()
plt.show()
```

## 🟧 Workflows no Orange Data Mining

Para ambos os exercícios, o fluxo de trabalho no Orange Data Mining segue uma estrutura similar, permitindo uma análise visual e interativa.

### Fluxo de Classificação (Solar)

1.  **File:** Carregar `SolarPrediction.csv`.
2.  **Feature Constructor:** Criar a variável `Radiation_Level` (ex: `1 if Radiation > [mediana] else 0`).
3.  **Select Columns:** Definir `Radiation_Level` como *Target* e as variáveis meteorológicas como *Features*.
4.  **Data Sampler:** Dividir os dados em 70% para treino e 30% para teste (com amostragem estratificada).
5.  **Models:** Conectar a amostra de treino aos widgets `Tree`, `Random Forest` e `SVM`.
6.  **Predictions:** Conectar os três modelos e os dados de teste.
7.  **Confusion Matrix:** Conectar o widget `Predictions` para visualizar e comparar a performance dos modelos.

### Fluxo de Regressão (Eólica)

1.  **File:** Carregar `T1.csv`.
2.  **Select Columns:** Definir `ActivePower_kW` como *Target* e `WindSpeed_ms` e `WindDirection_Degrees` como *Features*.
3.  **Data Sampler:** Dividir os dados em 80% para treino e 20% para teste.
4.  **Models:** Conectar a amostra de treino aos widgets `Linear Regression`, `Tree` e `Random Forest`.
5.  **Predictions:** Conectar os modelos e os dados de teste.
6.  **Evaluation:** Analisar as métricas (RMSE, R²) diretamente no widget `Predictions` ou usar um `Scatter Plot` para comparar visualmente os valores reais e previstos.
