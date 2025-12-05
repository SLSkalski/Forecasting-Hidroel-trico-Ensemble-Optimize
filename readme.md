# Previsão de Geração Hidrelétrica com Ensemble de Modelos (LightGBM, XGBoost, CatBoost)

## Visão Geral

Este projeto demonstra a construção de um pipeline completo para previsão da geração de energia hidrelétrica (em MW) utilizando um ensemble de modelos de Machine Learning (LightGBM, XGBoost e CatBoost). O objetivo é prever a geração com um horizonte de `X` dias à frente, incorporando engenharia de features temporais, lags e estatísticas móveis, além de análise de interpretabilidade com SHAP.

## Funcionalidades

- **Engenharia de Features:** Criação de features temporais (dia da semana, mês, etc.), lags do target, estatísticas móveis (médias, desvios) e features de variáveis externas (vazão, precipitação, nível de reservatório).
- **Split Temporal:** Divisão dos dados em treino e teste respeitando a ordem temporal para simular um cenário real de previsão.
- **Modelos Individuais:** Treinamento e avaliação de LightGBM, XGBoost e CatBoost Regressors.
- **Ensemble Ponderado:** Combinação das previsões dos modelos individuais através de uma média ponderada, com otimização dos pesos para melhorar a performance.
- **Avaliação Detalhada:** Cálculo de métricas de erro (MAE, RMSE, R², MAPE) para o ensemble e para cada modelo individual.
- **Visualizações:** Gráficos comparativos de previsões, métricas, resíduos e scatter plots para análise da qualidade das previsões.
- **Análise SHAP:** Interpretabilidade dos modelos utilizando SHAP (SHapley Additive exPlanations) para entender a importância e o impacto de cada feature nas previsões do ensemble e dos modelos individuais.
- **Salvamento de Modelos:** Persistência dos modelos treinados para uso futuro.

## Estrutura do Projeto

O código é organizado em funções modulares para facilitar a reutilização e manutenção:

- `create_time_features()`: Cria features baseadas na data.
- `create_lag_features()`: Cria features de lags do target.
- `create_rolling_features()`: Cria features de estatísticas móveis.
- `create_external_features()`: Cria lags e rolling para variáveis externas.
- `prepare_features()`: Pipeline completo de engenharia de features.
- `split_train_test()`: Realiza o split temporal dos dados.
- `train_lightgbm()`, `train_xgboost()`, `train_catboost()`: Funções para treinamento dos modelos individuais.
- `EnsembleModel` (Classe): Implementa o ensemble com métodos `train()`, `predict()` e `optimize_weights()`.
- `evaluate_ensemble()`: Avalia o desempenho do ensemble e dos modelos individuais.
- `plot_ensemble_comparison()`, `plot_residuals_comparison()`: Funções de visualização.
- `analyze_shap_ensemble()`: Realiza a análise SHAP para o ensemble.
- `exemplo_completo()`: Função que orquestra todo o pipeline (simulação de dados, engenharia de features, treinamento, otimização, avaliação e SHAP).

## Instalação

Certifique-se de ter as seguintes bibliotecas instaladas. Você pode instalá-las via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost catboost shap
Uso
Para executar o pipeline completo e ver um exemplo prático, basta chamar a função exemplo_completo().

if __name__ == "__main__":
    ensemble, df, results = exemplo_completo()
Parâmetros Configuráveis (dentro de exemplo_completo()):
FORECAST_HORIZON: Número de dias à frente que se deseja prever (padrão: 7).
Parâmetros dos modelos individuais (train_lightgbm, train_xgboost, train_catboost).
Avaliação dos Modelos
O pipeline imprime as métricas de avaliação (MAE, RMSE, R², MAPE) para o ensemble e para cada modelo individual, permitindo uma comparação direta do desempenho.

Exemplo de Resultados (simulados)
Ensemble:
  MAE:  81.16 MW
  RMSE: 99.49 MW
  R²:   0.9239
  MAPE: 4.37%

LightGBM:
  MAE:  82.40 MW
  RMSE: 102.77 MW
  R²:   0.9188
  MAPE: 4.47%

XGBoost:
  MAE:  82.89 MW
  RMSE: 101.09 MW
  R²:   0.9214
  MAPE: 4.47%

CatBoost:
  MAE:  81.43 MW
  RMSE: 100.67 MW
  R²:   0.9221
  MAPE: 4.37%
Análise SHAP
Após a avaliação, a função analyze_shap_ensemble gera plots de resumo SHAP para o ensemble e cada modelo, visualizando a importância e o impacto das features nas previsões.

Modelos Salvos
Os modelos treinados (LightGBM, XGBoost, CatBoost) são salvos nos formatos txt, json e cbm respectivamente, permitindo o carregamento e uso posterior para novas previsões.

ensemble_lightgbm.txt
ensemble_xgboost.json
ensemble_catboost.cbm

Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

Licença
Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes. ```