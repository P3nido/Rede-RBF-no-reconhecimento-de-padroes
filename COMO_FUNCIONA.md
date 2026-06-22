# Rede RBF no Reconhecimento de Padrões — Lógica do Projeto

Este documento explica **como o projeto funciona por dentro**: a teoria da Rede de
Base Radial (RBF), como cada etapa do enunciado foi traduzida em código e por que
cada decisão foi tomada. Todo o código está em `main.py` e roda com um único
comando:

```bash
python main.py
```

---

## 1. O problema

A rede deve verificar a **presença de radiação** em substâncias nucleares a partir
de duas variáveis características da radiação, `x1` e `x2`. Cada amostra tem uma
saída desejada `d`:

| `d`  | Significado            | Classe     |
|:----:|:-----------------------|:-----------|
| `+1` | radiação **Existente** | positiva   |
| `-1` | radiação **Inexistente** | negativa |

- **Treinamento:** 40 amostras conhecidas (`PP05_dados_treinamento.txt`)
- **Validação:** 10 amostras (`PP05_dados_validacao.txt`)

> O arquivo de validação só contém `x1` e `x2`. As saídas desejadas `d` da validação
> estão na Tabela 3 do enunciado (PDF) e foram embutidas no código na constante
> `D_VALIDACAO`.

---

## 2. Arquitetura da rede RBF

A rede tem **três camadas**, conforme a Figura 1 do enunciado:

```
                 camada escondida
                 (base radial)
   x1 ─────►  ( g1 )  gaussiana ─┐
        ╲   ╱                     ├─►  ( Σ )  saída linear ─► y
        ╳                         │
        ╱   ╲                     │
   x2 ─────►  ( g2 )  gaussiana ─┘
```

1. **Entrada:** as duas variáveis `x1`, `x2`.
2. **Camada escondida:** 2 neurônios com **ativação gaussiana** (base radial). Cada
   neurônio tem um *centro* `c_j` e uma *variância* `σ²_j`. Ele mede o quão perto a
   amostra está do seu centro.
3. **Saída:** 1 neurônio com **ativação linear** que combina as duas gaussianas.

O ponto central da RBF é que ela tem **dois estágios de treinamento totalmente
diferentes**:

| Estágio | Camada      | Técnica                       | É supervisionado? |
|:-------:|:------------|:------------------------------|:------------------|
| 1º      | escondida   | **k-means** (agrupamento)     | não (usa só `x`)  |
| 2º      | saída       | **regra Delta** (LMS)         | sim (usa `d`)     |

---

## 3. Estágio 1 — Camada escondida via *k-means* (Questão 1)

### Ideia
A camada escondida não é treinada por gradiente. Em vez disso, usamos o **k-means**
para descobrir onde estão os "grupos" naturais dos dados e colocamos um neurônio
gaussiano no centro de cada grupo.

### O que o enunciado pede
Computar os centros de **2 clusters** considerando **apenas os padrões com presença
de radiação** (`d = +1`). Ou seja, primeiro filtramos as amostras:

```python
mascara_radiacao = d_train == 1.0     # só os padrões com radiação
X_rad = X_train[mascara_radiacao]     # 19 das 40 amostras
```

### Como o k-means funciona (`kmeans`)
1. Sorteia 2 amostras como centros iniciais.
2. **Atribuição:** cada amostra vai para o centro mais próximo (distância euclidiana).
3. **Atualização:** cada centro vira a média das amostras que caíram nele.
4. Repete 2–3 até os centros pararem de se mover.

```python
distancias = np.linalg.norm(X[:, None, :] - centroides[None, :, :], axis=2)
novos_rotulos = np.argmin(distancias, axis=1)          # passo 2
novos_centroides = [X[rotulos == j].mean(axis=0) ...]  # passo 3
```

Para que "Cluster 1" e "Cluster 2" sejam sempre os mesmos entre execuções, os
clusters são **ordenados pelo `x1` do centro** ao final.

### A variância (`variancia_cluster`)
Define a "abertura" da gaussiana. Usamos a convenção do livro (Silva et al.):

```
σ²_j = (1 / n_j) · Σ ‖ x − c_j ‖²        (x nas amostras do cluster j)
```

ou seja, a **média das distâncias quadráticas** das amostras do cluster até o seu
centro.

### Resultado (Tabela 1)

| Cluster | Centro (x1, x2)        | Variância | Nº padrões |
|:-------:|:-----------------------|:---------:|:----------:|
| 1       | (0.164833, 0.612117)   | 0.029806  | 6          |
| 2       | (0.398969, 0.157131)   | 0.038460  | 13         |

> **Validação:** o k-means convergiu para a mesma solução em 50 sementes diferentes,
> e os valores batem exatamente com o `scikit-learn.KMeans`.

---

## 4. A ativação gaussiana (`ativacao_gaussiana`)

Antes do 2º estágio, toda amostra precisa ser transformada na saída da camada
escondida. Cada neurônio `j` responde:

```
g_j(x) = exp( − ‖ x − c_j ‖² / (2 · σ²_j) )
```

Interpretação: `g_j` vale **≈ 1** quando a amostra está em cima do centro `c_j` e
**→ 0** quando está longe. Assim, cada amostra de entrada `(x1, x2)` é convertida em
um par `(g1, g2)` — o quão perto ela está de cada um dos dois grupos de radiação.

---

## 5. Estágio 2 — Camada de saída via regra Delta (Questões 2 e 3)

### Ideia
Agora sim há aprendizado supervisionado: ajustamos os pesos do neurônio de saída
para que `y` se aproxime de `d`. Como a saída é **linear**, o problema é idêntico ao
de um ADALINE e é resolvido pela **regra Delta (LMS)**.

### O modelo da saída (`saida_rbf`)
```
y = W(2)_1,1 · g1 + W(2)_2,1 · g2 − θ
```

No código, o limiar `θ` é tratado como um peso sobre uma entrada fixa `−1`, então a
entrada do neurônio é o vetor aumentado `[ −1, g1, g2 ]` e os pesos são
`[ θ, W1, W2 ]`.

### O treinamento (`treinar_camada_saida`)
Para cada amostra, calcula a saída e corrige os pesos na direção do erro:

```python
u = G_aug[k] @ w               # saída linear da amostra k
w = w + eta * (d[k] - u) * G_aug[k]   # regra Delta
```

- **Taxa de aprendizado** `η = 0,01` (do enunciado).
- **Critério de parada:** quando o erro quadrático médio quase não muda mais entre
  duas épocas: `|EQM(atual) − EQM(anterior)| ≤ ε`, com `ε = 10⁻⁷`.

O **EQM** (erro quadrático médio) por época é:

```
EQM = (1/p) · Σ_k ½ · (d(k) − u(k))²
```

### Resultado (Tabela 2)

| Parâmetro   | Valor    |
|:-----------:|:--------:|
| W(2)_1,1    | 2.375113 |
| W(2)_2,1    | 2.696487 |
| θ₁          | 1.001938 |

Convergiu em **336 épocas**, EQM final = **0.117121**.

> **Validação:** como a saída é linear, o mínimo do EQM é único (problema convexo).
> Os pesos encontrados pela regra Delta batem com a solução analítica de mínimos
> quadrados (pseudo-inversa) e são idênticos para qualquer semente de inicialização.

### Questão 3 — gráfico do EQM
A função `salvar_grafico_eqm` traça o EQM em função de cada época. A curva cai de
forma **monotônica** (sem oscilação), o que confirma que `η = 0,01` é adequado e que
o treinamento de fato convergiu. Os valores de cada época também são exportados para
`EQM_por_epoca.xlsx`.

---

## 6. Validação e pós-processamento (Questão 4)

A saída `y` é contínua, mas a decisão é binária. Aplica-se o **pós-processamento** do
enunciado (`pos_processar`):

```
y_post = +1   se  y ≥ 0
y_post = −1   se  y < 0
```

Para as 10 amostras de validação, comparamos `y_post` com `d` e calculamos a taxa de
acertos. Tudo é gravado em `Tabela3_validacao.xlsx`, com cada linha colorida de verde
(acerto) ou vermelho (erro).

**Resultado: 8/10 acertos = 80%.** Os 2 erros são padrões com radiação (`d=+1`)
classificados como `−1`, ambos perto da fronteira (`y` ligeiramente negativo).

---

## 7. Matriz de confusão e métricas (Questão 5)

Como o problema é binário, a matriz de confusão é **2×2** (`matriz_confusao_binaria`),
com a classe positiva = radiação Existente (`+1`):

|                    | Predito +1 | Predito −1 |
|:-------------------|:----------:|:----------:|
| **Real +1**        | TP = 2     | FN = 2     |
| **Real −1**        | FP = 0     | TN = 6     |

A função `parametros_classificacao` calcula a partir dela:

| Métrica         | Fórmula                | Valor    |
|:----------------|:-----------------------|:--------:|
| N_acertos       | TP + TN                | 8        |
| N_erros         | FP + FN                | 2        |
| Acurácia        | (TP+TN)/total          | 80,00 %  |
| Sensibilidade   | TP/(TP+FN)             | 50,00 %  |
| Especificidade  | TN/(TN+FP)             | 100,00 % |
| Precisão        | TP/(TP+FP)             | 100,00 % |

### Leitura do resultado
A rede **nunca dá falso alarme** (FP = 0 → precisão e especificidade de 100%), mas
**deixa passar metade dos casos reais de radiação** (sensibilidade de 50%). Em
detecção de radiação, o FN (não detectar radiação existente) é o erro mais perigoso.

### Estratégias para melhorar
A causa é estrutural: só há **2 centros gaussianos** sobre a região `d=+1`, e os FN
são padrões com radiação distantes desses centros. Para melhorar:

1. **Mais neurônios ocultos / clusters** (k > 2) → cobre melhor a região de radiação.
2. **Alargar as gaussianas** (σ maior) → cada neurônio "enxerga" amostras mais longe.
3. **Mais dados de treinamento** na região mal coberta.
4. **Centros guiados pelas duas classes**, não só por `d=+1`.

---

## 8. Mapa do código (`main.py`)

| Função                       | Papel                                              |
|:-----------------------------|:---------------------------------------------------|
| `carregar_treinamento` / `carregar_validacao` | leem os `.txt` (ignorando cabeçalho) |
| `kmeans`                     | Q1 — agrupa os padrões `d=+1` em 2 clusters        |
| `variancia_cluster`          | Q1 — variância (abertura) de cada gaussiana        |
| `ativacao_gaussiana`         | converte `(x1,x2)` em `(g1,g2)`                    |
| `treinar_camada_saida`       | Q2 — regra Delta (LMS) para os pesos da saída      |
| `saida_rbf`                  | calcula a resposta `y` da rede                     |
| `pos_processar`              | Q4 — `y_post = ±1`                                 |
| `matriz_confusao_binaria`    | Q5 — matriz 2×2                                    |
| `parametros_classificacao`   | Q5 — acurácia, sensibilidade, etc.                 |
| `gerar_planilha_*`           | exportam as Tabelas 1, 2, 3 e métricas para `.xlsx`|
| `salvar_grafico_*`           | geram os gráficos em `graphics/`                   |
| `main`                       | executa as questões 1 → 5 em sequência             |

### Fluxo geral
```
ler dados ─► [Q1] k-means (centros, variâncias)
                     │
                     ▼
          ativação gaussiana de todas as amostras
                     │
                     ▼
          [Q2] regra Delta ─► pesos da saída ─► [Q3] gráfico do EQM
                     │
                     ▼
          [Q4] validação + pós-processamento ─► taxa de acertos
                     │
                     ▼
          [Q5] matriz de confusão + métricas
```

---

## 9. Arquivos de saída

| Questão | Arquivos gerados                                             |
|:-------:|:-------------------------------------------------------------|
| Q1      | `Tabela1_clusters.xlsx`, `graphics/q1_clusters_kmeans.png`   |
| Q2      | `Tabela2_pesos.xlsx`                                          |
| Q3      | `graphics/q3_eqm_etapa2.png`, `EQM_por_epoca.xlsx`           |
| Q4      | `Tabela3_validacao.xlsx`                                      |
| Q5      | `Tabela_Q5_metricas.xlsx`, `graphics/q5_matriz_confusao.png` |
