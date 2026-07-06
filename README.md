# Agrupamentos de Tipos de Setores Censitários da Bahia via Aprendizado Não Supervisionado

**Disciplina:** Tópicos em Inteligência Artificial Aprendizado Não Supervisionado
**Dataset:** CNEFE 2010 e Censo 2022 (IBGE) Estado da Bahia
**Algoritmo principal:** UMAP → HDBSCAN

---

## Sumário

1. [Contexto e Motivação](#1-contexto-e-motivação)
2. [Por que Aprendizado Não Supervisionado](#2-por-que-aprendizado-não-supervisionado)
3. [Dados e Infraestrutura](#3-dados-e-infraestrutura)
   - [3.4 Coleta e Pré-processamento](#34-coleta-e-pré-processamento)
4. [Engenharia de Features](#4-engenharia-de-features)
5. [Pipeline de Modelagem 2022](#5-pipeline-de-modelagem-2022)
6. [Análise de 2010: Validação a Posteriori e Consistência Temporal](#6-análise-de-2010-validação-a-posteriori-e-consistência-temporal)
7. [Análise Temporal 2010 → 2022](#7-análise-temporal-2010--2022)
8. [Comparação de Algoritmos: DBSCAN vs HDBSCAN](#8-comparação-de-algoritmos-dbscan-vs-hdbscan)
9. [Resultados e Interpretação](#9-resultados-e-interpretação)
10. [Aderência aos Requisitos do Projeto](#10-aderência-aos-requisitos-do-projeto)
11. [Estrutura do Repositório](#11-estrutura-do-repositório)
12. [Perguntas Frequentes sobre a Metodologia](#12-perguntas-frequentes-sobre-a-metodologia)
13. [Referências Bibliográficas](#13-referências-bibliográficas)

---

## 1. Contexto e Motivação

O Instituto Brasileiro de Geografia e Estatística (IBGE) organiza o território nacional em **setores censitários**  unidades mínimas de coleta de dados, cada uma com aproximadamente 200 a 400 domicílios. No estado da Bahia, são mais de 30.000 setores, cobrindo desde o centro histórico de Salvador até comunidades rurais isoladas no sertão.

O IBGE classifica cada setor de forma binária: **urbano** ou **rural**. Essa divisão serve à coleta, mas é insuficiente para qualquer análise de uso do solo com granularidade real. Um hospital, uma comunidade, um condomínio fechado e um conjunto habitacional popular são todos classificados como "urbanos"  mas têm perfis radicalmente distintos de uso, densidade e serviços.

**O problema prático:** gestores de saúde pública, urbanistas e pesquisadores precisam de tipologias mais finas para tomar decisões. Onde estão concentrados os estabelecimentos de saúde? Quais setores estão em processo de transição rural→urbano? Onde há expansão de domicílios coletivos (hospitais, presídios, asilos)?

**A pergunta de pesquisa:** *É possível descobrir automaticamente tipologias significativas de setores censitários a partir dos dados brutos de endereçamento, sem usar nenhum rótulo pré-definido?*

---

## 2. Por que Aprendizado Não Supervisionado

A primeira ideia que nos vem a mente é treinar um classificador supervisionado usando o campo `situacao` (urbano/rural) do CNEFE 2010 como label. Mas isso enfrenta dois problemas fundamentais.

### 2.1 Incompatibilidade de Schema entre os Censos

O CNEFE 2010 e o Censo 2022 usam schemas completamente diferentes para descrever endereços:

| Campo         | CNEFE 2010                         | Censo 2022                                      |
| ------------- | ---------------------------------- | ----------------------------------------------- |
| Tipo de local | `tipo` (RUA, FAZENDA, ESTRADA…) | `COD_ESPECIE` (1–8, tipo de estabelecimento) |
| Situação    | `situacao` (1=urbano, 2=rural)   | não existe como campo separado                 |
| Coordenadas   | **não tem**                 | `LATITUDE`, `LONGITUDE` por endereço       |

Não há mapeamento direto entre `tipo` do CNEFE 2010 e `COD_ESPECIE` do Censo 2022. Treinar um modelo no schema de 2010 e aplicar em 2022 exigiria uma cadeia de transformações com suposições arbitrárias, comprometendo a validade da comparação temporal.

### 2.2 O Rótulo Binário é Insuficiente

Mesmo que os schemas fossem compatíveis, o objetivo não é replicar a classificação urbano/rural é descobrir os subtipos *dentro* de cada classe. Uma abordagem supervisionada reproduziria exatamente o que o IBGE já entrega. O valor científico está em revelar estrutura latente que a classificação oficial não captura.

### 2.3 O Argumento Central

Aprendizado não supervisionado não é a segunda opção aqui  é a única abordagem que:

- Funciona nos dois schemas sem adaptação manual
- Permite comparação temporal genuína (a estrutura dos dados fala por si)
- Pode revelar tipologias que não existem como categorias oficiais

A validação *a posteriori*  cruzar os clusters com o campo `situacao` que deliberadamente não foi usado como feature  serve como teste de sanidade: se o algoritmo recuperar a estrutura urbano/rural sem ter visto esse campo, está capturando algo real nos dados.

---

## 3. Dados e Infraestrutura

### 3.1 Fontes de Dados

**CNEFE 2010** (`data/cnefe_2010/`)
Cadastro Nacional de Endereços para Fins Estatísticos  Censo 2010. Para a Bahia: ~4,8 milhões de registros de endereço. Campos principais: `setor`, `uf`, `situacao` (1=urbano, 2=rural), `tipo` (tipo de logradouro). Sem coordenadas geográficas por endereço.

**Censo 2022** (`data/censo 2022/29_BA.parquet`)
CNEFE do Censo Demográfico 2022  Bahia. ~9 milhões de registros. Campos principais: `COD_SETOR`, `COD_ESPECIE` (tipo de estabelecimento), `COD_INDICADOR_FINALIDADE_CONST` (finalidade da construção), `LATITUDE`, `LONGITUDE`.

![Dispersão geográfica dos endereços do Censo 2022 na Bahia (amostra 50k)](outputs/figures/censo_dispersao_geo.png)

*Esta figura usa uma amostra de 50 mil endereços exclusivamente para fins de visualização, via `USING SAMPLE 50000` no DuckDB. Todo o restante do pipeline, incluindo o cálculo de proporções, a normalização, o UMAP e o HDBSCAN, opera sobre os 9 milhões de registros completos do Censo 2022. A amostragem aqui serve apenas para tornar o scatter plot renderizável: plotar 9 milhões de pontos sobrecarregaria qualquer ambiente gráfico sem acrescentar informação visual relevante.*

### 3.2 Por que Parquet

Com ~9 milhões de linhas apenas para a Bahia, o formato CSV seria inviável: o arquivo seria >2 GB e o carregamento em memória via `pd.read_csv()` esgotaria a RAM de uma máquina de desenvolvimento comum.

O formato **Parquet** resolve isso de três formas:

1. **Armazenamento colunar**: lê apenas as colunas necessárias, não o arquivo inteiro
2. **Compressão nativa**: o arquivo é ~8x menor que o CSV equivalente
3. **Tipagem forte**: tipos de dados preservados sem parsing, eliminando erros de conversão

O CNEFE 2010 está particionado em múltiplos arquivos `.snappy.parquet`  padrão eficiente para dados grandes que permite leitura paralela e acesso seletivo por partição.

### 3.3 Por que DuckDB

Com Parquet, ainda seria necessário carregar o arquivo inteiro em memória para fazer a agregação por setor. O **DuckDB** resolve isso executando SQL diretamente sobre os arquivos Parquet sem carregamento prévio:

```python
import duckdb

con = duckdb.connect()
CENSO = "'../data/censo 2022/29_BA.parquet'"

df_setores = con.execute(f"""
    SELECT
        COD_SETOR,
        COUNT(*) AS total_enderecos,
        AVG(LATITUDE)  AS lat_centroide,
        AVG(LONGITUDE) AS lon_centroide,
        SUM(CASE WHEN COD_ESPECIE = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS prop_domicilio_particular,
        -- demais proporções...
    FROM read_parquet({CENSO})
    WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL
    GROUP BY COD_SETOR
    HAVING COUNT(*) >= 5
""").df()
```

Essa query reduz 9 milhões de linhas para ~30.000 setores em segundos, sem ocupar memória com os dados brutos. O resultado já chega como `DataFrame` Pandas, pronto para o pipeline de ML.

**Por que 9 milhões de registros foram processados em tempo hábil?**

Quatro fatores combinados explicam o desempenho:

1. **Leitura seletiva de colunas (Parquet):** o arquivo tem 34 colunas, mas a query lê apenas 4 (`COD_SETOR`, `COD_ESPECIE`, `LATITUDE`, `LONGITUDE`). O formato colunar do Parquet permite isso nativamente: os dados das colunas não selecionadas nunca saem do disco, reduzindo o volume de I/O em cerca de 90%.
2. **Execução SQL embutida (DuckDB):** a agregação `GROUP BY COD_SETOR` é executada dentro do próprio DuckDB, diretamente sobre o arquivo Parquet. Os 9 milhões de linhas nunca são carregados no Python: o que entra na memória é apenas o resultado agregado, com ~30.000 linhas.
3. **Redução de escala antecipada:** a query transforma o problema de ML antes mesmo de qualquer algoritmo ser chamado. Os modelos (UMAP, HDBSCAN, DBSCAN) nunca veem os 9 milhões de endereços, apenas os 30.355 vetores de proporção resultantes.
4. **UMAP e HDBSCAN em escala reduzida:** com 30.355 pontos em 11 dimensões, o UMAP leva poucos minutos em CPU. O HDBSCAN aplicado no espaço 2D resultante é quase instantâneo. A arquitetura do pipeline garante que a complexidade computacional dos algoritmos mais custosos recaia sobre um dataset já compacto, não sobre os dados brutos.

### 3.4 Coleta e Pré-processamento

#### 3.4.1 Obtenção dos Dados

Os dados foram obtidos diretamente do portal do IBGE:

- **CNEFE 2010**: baixado em formato de tabelas DBF/CSV por Unidade da Federação e convertido para Parquet com compressão Snappy. O arquivo da Bahia contém ~4,8 milhões de registros de endereço distribuídos em múltiplos arquivos particionados.
- **Censo 2022**: disponibilizado pelo IBGE em formato CSV por UF e convertido para Parquet antes do início da análise. O arquivo `29_BA.parquet` resultante contém ~9 milhões de registros com coordenadas geográficas por endereço, inovação ausente no CNEFE 2010.

#### 3.4.2 Limpeza e Filtragem

Antes da agregação, dois filtros são aplicados diretamente na query DuckDB:

```python
WHERE LATITUDE IS NOT NULL AND LONGITUDE IS NOT NULL   # remove endereços sem geocodificação
HAVING COUNT(*) >= 5                                   # remove setores com menos de 5 endereços
```

O filtro de coordenadas nulas elimina endereços com nível de geocodificação insuficiente (`NV_GEO_COORD >= 4`). O filtro de mínimo de endereços por setor garante que as proporções calculadas não sejam dominadas por acaso estatístico: um setor com 2 endereços em que 1 é um estabelecimento de saúde teria `prop_estab_saude = 0.5`, valor artificialmente alto sem significado real. Com `HAVING COUNT(*) >= 5`, esse tipo de setor é excluído antes da modelagem.

Após os filtros: **30.355 setores** do Censo 2022 e **23.763 setores** do CNEFE 2010 compõem o conjunto final.

#### 3.4.3 Transformação em Vetor de Proporções

A etapa central do pré-processamento é a transformação de 9 milhões de linhas de endereço em 30.355 vetores de proporção por setor. Toda essa transformação ocorre em SQL, sem carregar os dados brutos em memória:

```python
# Cada CASE WHEN calcula a proporção de endereços de cada tipo no setor
SUM(CASE WHEN COD_ESPECIE = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS prop_domicilio_particular,
SUM(CASE WHEN COD_ESPECIE = 2 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS prop_domicilio_coletivo,
SUM(CASE WHEN COD_ESPECIE = 3 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS prop_estab_agropecuario,
# ... demais 8 proporções
AVG(LATITUDE)  AS lat_centroide,   # centroide geográfico do setor
AVG(LONGITUDE) AS lon_centroide
```

O resultado intermediário é salvo em `outputs/setores_features.parquet` (30.355 × 15 colunas), servindo como ponto de entrada para todas as etapas seguintes sem necessidade de re-processar os dados brutos.

#### 3.4.4 Normalização

Com as proporções calculadas, aplicamos `StandardScaler` (z-score) para colocar todas as features na mesma escala antes do UMAP e do clustering:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df_setores[FEATURES])   # ajusta e transforma
```

O objeto `scaler` é preservado em memória para ser reutilizado com `scaler.transform()` (sem re-ajuste) nos experimentos de comparação temporal e COM/SEM coordenadas geográficas, garantindo que as escalas sejam comparáveis entre as configurações.

#### 3.4.5 Resumo do Pipeline

```
Dados brutos (IBGE)
    │
    ├── CNEFE 2010 (~4,8M linhas, CSV/DBF → Parquet)
    │       └── DuckDB: agrega por setor → 23.763 setores × 12 proporções
    │
    └── Censo 2022 (~9M linhas, CSV → Parquet)
            └── DuckDB: agrega por setor → 30.355 setores × 11 proporções + lat/lon
                    │
                    ├── StandardScaler → z-scores (X)
                    │
                    ├── UMAP (n_components=2) → X_umap
                    │
                    └── HDBSCAN (min_cluster_size=50) → 13 clusters
                                │
                                └── outputs/setores_clusterizados.parquet
```

---

## 4. Engenharia de Features

### 4.1 A Ideia Central

Cada setor censitário recebe um **vetor de proporções** que descreve sua composição funcional. Em vez de contar endereços em valor absoluto (o que dependeria do tamanho do setor), calculamos a fração de cada tipo de estabelecimento sobre o total de endereços do setor.

Isso tem uma propriedade importante: dois setores de tamanhos muito diferentes, mas com a mesma composição funcional, terão vetores próximos no espaço de features. O algoritmo compara perfis, não volumes.

### 4.2 Features do Censo 2022

O campo `COD_ESPECIE` do CNEFE 2022 classifica cada endereço em 8 categorias definidas pelo IBGE:

| COD_ESPECIE | Descrição IBGE                       | Feature criada                |
| ----------- | -------------------------------------- | ----------------------------- |
| 1           | Domicílio Particular                  | `prop_domicilio_particular` |
| 2           | Domicílio Coletivo                    | `prop_domicilio_coletivo`   |
| 3           | Estabelecimento Agropecuário          | `prop_estab_agropecuario`   |
| 4           | Estabelecimento de Ensino              | `prop_estab_ensino`         |
| 5           | Estabelecimento de Saúde              | `prop_estab_saude`          |
| 6           | Estabelecimento com Outras Finalidades | `prop_estab_outras`         |
| 7           | Em Construção                        | `prop_construcao`           |
| 8           | Estabelecimento Religioso              | `prop_estab_religioso`      |

Adicionalmente, `COD_INDICADOR_FINALIDADE_CONST` indica a finalidade declarada da construção, gerando 3 features complementares: `prop_finalidade_residencial`, `prop_finalidade_comercial`, `prop_finalidade_mista`.

**Total: 11 features** para o Censo 2022.

Essa lista é o contrato central do pipeline: todo passo seguinte normalização, UMAP e HDBSCAN recebe exatamente `df[FEATURES].values` como entrada. A ordem importa para o heatmap de z-scores (seções 5 e 9).

```python
FEATURES = [
    'prop_domicilio_particular', 'prop_domicilio_coletivo',
    'prop_estab_agropecuario', 'prop_estab_ensino', 'prop_estab_saude',
    'prop_estab_outras', 'prop_construcao', 'prop_estab_religioso',
    'prop_finalidade_residencial', 'prop_finalidade_comercial', 'prop_finalidade_mista'
]
```

### 4.3 Features do CNEFE 2010

O CNEFE 2010 não possui `COD_ESPECIE`. A informação disponível é o campo `tipo`, que descreve o tipo de logradouro. Calculamos as proporções dos 12 tipos mais frequentes na Bahia. O agrupamento comentado no código (urbanos/rurais) é apenas descritivo o algoritmo não vê essa separação, ela serve para o leitor entender a semântica intuitiva de cada feature:

```python
FEATURES_2010 = [
    # logradouros tipicamente urbanos
    'prop_rua', 'prop_avenida', 'prop_travessa', 'prop_praca', 'prop_alameda', 'prop_beco',
    # logradouros tipicamente rurais
    'prop_fazenda', 'prop_estrada', 'prop_caminho', 'prop_rodovia', 'prop_sitio', 'prop_povoado'
]
```

**Decisão metodológica crítica:** o CNEFE 2010 também possui `prop_urbano` e `prop_rural` (derivadas do campo `situacao`). Essas features foram **deliberadamente excluídas** do clustering. Incluí-las tornaria a análise circular estaríamos encontrando clusters de "urbano" e "rural" usando exatamente o campo que define urbano e rural. O objetivo é descobrir estrutura com base no tipo de uso do solo, e depois verificar se essa estrutura corresponde à classificação oficial.

### 4.4 Normalização

Todas as features passam por `StandardScaler` (z-score). `fit_transform()` em uma única chamada ajusta a média e desvio-padrão de cada coluna nos dados de treino e já aplica a transformação equivalente a calcular `(x - média) / desvio_padrão` para cada feature. O objeto `scaler` é salvo em memória pois será reutilizado no experimento COM/SEM geo (seção 6.3) com `scaler.transform()`, sem re-ajuste:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df_setores[FEATURES])
```

As proporções variam de forma muito diferente: `prop_domicilio_particular` fica próxima de 1.0 na maioria dos setores urbanos, enquanto `prop_domicilio_coletivo` raramente passa de 0.05. Sem normalização, a feature dominante controlaria sozinha o espaço de distâncias.

O resultado final é salvo em `outputs/setores_features.parquet`: **30.355 setores × 11 features** (z-scores).

![Distribuição do total de endereços por setor censitário](outputs/figures/distribuicao_enderecos_setor.png)

*Histograma do número de endereços por setor após o filtro `HAVING COUNT(*) >= 5`. A distribuição é assimétrica à direita: a maioria dos setores tem entre 50 e 400 endereços (faixa esperada pelo IBGE), mas há setores com milhares (áreas institucionais grandes ou setores de alta densidade urbana). Essa variabilidade justifica o uso de proporções em vez de contagens absolutas como features.*

---

## 5. Pipeline de Modelagem 2022

### 5.1 O Problema de Clustering em Alta Dimensão

> **Nota sobre dimensionalidade:** os valores "11D" nesta seção referem-se exclusivamente ao **Censo 2022**, que possui 11 features de proporção (8 categorias de `COD_ESPECIE` + 3 indicadores de finalidade). O CNEFE 2010, tratado na seção 6, tem 12 features de tipo de logradouro e portanto opera em 12D. Os dois datasets têm dimensionalidades distintas porque seus campos originais são diferentes.

Algoritmos baseados em densidade sofrem com a **maldição da dimensionalidade** em espaços de alta dimensão. Em 11 dimensões, a distância entre pontos tende a se homogeneizar: pontos próximos e distantes ficam numericamente semelhantes, e a noção de "vizinhança densa" colapsa.

A tentativa direta de HDBSCAN nos 11 features do Censo 2022 confirma isso:

```
                           11D direto    UMAP→HDBSCAN
Clusters encontrados              7              13
Ruído (%)                      89.3%           0.3%
```

Com HDBSCAN direto em 11D, **89% dos setores são descartados como ruído**. O algoritmo não encontra estrutura densa suficiente no espaço original.

### 5.2 UMAP como Pré-processamento

**UMAP** (Uniform Manifold Approximation and Projection) reduz as 11 dimensões para 2, preservando a estrutura de vizinhança local. Se dois setores são similares em composição funcional, eles ficam próximos no espaço 2D do UMAP.

#### A redução: de 11D para 2D

O dado bruto de entrada tem 11 features (as proporções por setor), logo cada setor é um ponto em um espaço de 11 dimensões. O UMAP projeta esses 30.355 pontos em apenas 2 coordenadas (`umap_x`, `umap_y`), com o objetivo de manter a vizinhança: se dois setores eram vizinhos próximos em 11D, devem continuar próximos em 2D.

**Por que 2D e não 3D?** A escolha de `n_components=2` tem duas vantagens práticas. Primeira: 2D é diretamente visualizável em um gráfico de dispersão, o que permite inspeção humana da estrutura antes e depois do clustering, e facilita a comunicação dos resultados. Segunda: para este conjunto de dados, a separação entre os grupos já é suficientemente clara em 2D grupos como Dom. Coletivo e Rural Agrícola aparecem como ilhas bem separadas na projeção, sem necessidade de uma terceira dimensão para distingui-los. O uso de 3D aumentaria levemente a informação preservada, mas tornaria o clustering e a validação visual mais custosos sem ganho substancial.

**O que é preservado e o que é perdido:** o UMAP preserva a *estrutura topológica local*  quem é vizinho de quem, mas não preserva distâncias absolutas nem geometria global. As coordenadas `umap_x` e `umap_y` não têm interpretação direta (não são componentes principais, não representam nenhuma feature original). O que importa é a posição relativa: dois setores próximos no espaço UMAP têm composição funcional similar; dois setores distantes têm composição diferente.

**Importante:** a projeção UMAP não é exclusiva do HDBSCAN. Na seção 8, o DBSCAN também é aplicado sobre as mesmas coordenadas `umap_x` e `umap_y`. Ambos os algoritmos recebem exatamente a mesma entrada 2D, tornando a comparação de resultados diretamente válida.

```python
import umap

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=30,   # vizinhança maior → preserva estrutura global
    min_dist=0.0,     # pontos similares ficam compactos → melhor para HDBSCAN
    random_state=42
)
X_umap = reducer.fit_transform(X)
```

Os parâmetros têm justificativa direta:

- `n_neighbors=30`: com 30.000 pontos, valor alto de vizinhança garante que a estrutura global (relação entre todos os tipos de setor) seja preservada, não apenas micro-estruturas locais
- `min_dist=0.0`: força pontos similares a se agruparem o máximo possível no espaço 2D, criando separação visual clara entre grupos, essencial para que o HDBSCAN encontre fronteiras nítidas

![Projeção UMAP dos 30.355 setores censitários da Bahia em 2D](outputs/figures/umap_setores.png)

*Cada ponto é um setor censitário posicionado no espaço 2D do UMAP. Setores com composição funcional similar ficam próximos. Os grupos visualmente separados já indicam a estrutura de clusters antes de qualquer algoritmo de agrupamento: ilhas compactas (domicílios coletivos, setores rurais agrícolas) contrastam com a grande nuvem central de setores residenciais urbanos.*

### 5.3 HDBSCAN no Espaço UMAP

Com o espaço reduzido a 2D, o HDBSCAN identifica regiões de alta densidade e as separa em clusters. `fit_predict()` retorna um array de rótulos inteiros: valores ≥ 0 indicam o cluster atribuído, e `-1` indica ruído (ponto que não pertence a nenhum cluster denso). Cada parâmetro controla um critério diferente de "o que é um cluster":

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,    # cluster deve ter ao menos 50 setores para existir
    min_samples=3,          # ponto precisa de 3 vizinhos próximos para ser considerado core
    metric='euclidean',
    cluster_selection_method='eom',  # Excess of Mass: prefere clusters compactos ao invés de hierárquicos
    prediction_data=True             # guarda estrutura interna para soft clustering futuro
)
labels = clusterer.fit_predict(X_umap)
```

**Resultado:** 13 clusters, 0.3% de ruído (90 setores).

![Single Linkage Tree: estrutura hierárquica do HDBSCAN (BA 2022)](outputs/figures/single_linkage_tree_ba.png)

*A árvore de condensação do HDBSCAN mostra como os clusters se formam à medida que o limiar de densidade aumenta. Cada galho representa um cluster persistindo em múltiplos níveis de densidade; galhos que duram mais são mais robustos. A espessura vertical de cada galho é proporcional ao número de setores. O algoritmo seleciona os 13 galhos mais persistentes como clusters finais, descartando os demais como ruído.*

### 5.4 Por que o Silhouette Score em 11D é Negativo

O Silhouette Score calculado no espaço 11D original é negativo para os clusters encontrados pelo UMAP→HDBSCAN, enquanto no espaço UMAP é positivo:

```
Silhouette UMAP (2D) :  0.3078
Silhouette 11D       : -0.0989
```

Isso não é um erro  é uma consequência da natureza do UMAP. Com `min_dist=0.0`, o UMAP forma clusters baseados em **vizinhança topológica**: dois setores estão no mesmo cluster se estão conectados por uma cadeia de vizinhos próximos, mesmo que sua distância euclidiana direta em 11D seja grande. O Silhouette mede distância euclidiana  logo, clusters topológicos podem parecer sobrepostos quando vistos em 11D.

O trade-off é deliberado:

- **11D direto** → clusters semanticamente puros, mas 89% de ruído (cobertura inutilizável para qualquer análise)
- **UMAP→HDBSCAN** → 0.3% de ruído, cobertura total, Silhouette positivo no espaço onde o clustering foi feito

### 5.5 Nomenclatura Semântica dos Clusters

Cada cluster recebe um nome baseado no perfil de z-scores médio de todos os seus setores, não apenas na feature de maior valor absoluto. Os nomes são atribuídos manualmente após inspeção visual do heatmap de z-scores (notebook 05): uma linha do heatmap com z-score muito positivo em `prop_domicilio_coletivo` e negativo em todas as outras, por exemplo, leva ao nome "Domicílio Coletivo (Institucional)". Não há automação nessa etapa é interpretação humana dos padrões quantitativos:

```python
NOMES_CLUSTER = {
    0:  'Residencial Puro (núcleo A)',
    1:  'Residencial Puro (núcleo B)',
    2:  'Residencial Puro (núcleo C)',
    3:  'Residencial Puro (núcleo D)',
    4:  'Residencial Puro',
    5:  'Residencial Denso',
    6:  'Rural / Peri-urbano Misto',
    7:  'Residencial c/ Educacional',
    8:  'Uso Misto Comercial',
    9:  'Rural (baixa densidade)',
    10: 'Rural Agrícola',
    11: 'Domicílio Coletivo (Institucional)',
    12: 'Setor Urbano Típico',
}
```

![Perfil z-score médio por cluster: Censo 2022 BA](outputs/figures/perfil_clusters_heatmap.png)

*Heatmap de z-scores médios: linhas são clusters, colunas são features. Cores quentes (positivo) indicam que o cluster tem proporção acima da média; cores frias (negativo), abaixo. A leitura é por linha: o cluster 11 (Dom. Coletivo) aparece como uma faixa isolada com z-score extremamente positivo em `prop_domicilio_coletivo` e negativo em todo o resto. Clusters rurais se destacam em `prop_estab_agropecuario`; clusters residenciais urbanos têm z-score alto em `prop_domicilio_particular` e `prop_finalidade_residencial`.*

Os clusters 0-3 merecem atenção: seus z-scores são praticamente idênticos em todas as 11 features. O UMAP os separou porque estão geograficamente distribuídos em cidades diferentes da Bahia (Salvador, Feira de Santana, Vitória da Conquista, Ilhéus). São o mesmo tipo funcional de setor em localizações distintas o UMAP capturou tanto a similaridade de composição quanto a separação geográfica residual.

![13 clusters individuais no mapa da Bahia: clusters 0-3 em cidades distintas](outputs/figures/clusters_mapa_13_individuais.png)

*Distribuição geográfica dos 13 clusters individuais. Os clusters 0-3 (Residencial Puro A-D) aparecem cada um concentrado em uma cidade diferente: Salvador, Feira de Santana, Vitória da Conquista e Ilhéus. Isso confirma que o UMAP com coordenadas geográficas capturou tanto a similaridade funcional quanto a separação espacial. O cluster 11 (Dom. Coletivo) aparece como pontos dispersos nas cidades polo.*

![Tipos semânticos agrupados no mapa da Bahia](outputs/figures/clusters_mapa_tipos.png)

*Visão de alto nível com clusters agrupados em 5 tipos semânticos: Residencial (azul), Rural (verde), Uso Misto (laranja), Dom. Coletivo (vermelho) e Comercial/Serviços (amarelo). A separação geográfica é clara: o interior e o sertão são predominantemente Rural; o litoral e as capitais regionais são Residencial; Dom. Coletivo aparece pontualmente nas cidades com hospitais e presídios de referência estadual.*

---

## 6. Análise de 2010: Validação a Posteriori e Consistência Temporal

O termo *validação cruzada* aqui não se refere ao k-fold cross-validation de aprendizado supervisionado. Não há divisão treino/teste, não há modelo sendo avaliado por generalização e não há rótulo sendo predito. O que se faz nesta seção é aplicar o mesmo pipeline não supervisionado ao CNEFE 2010 de forma independente e, somente após o clustering, cruzar o resultado com o campo `situacao` (urbano/rural) que não foi usado em nenhum momento como feature. Esse cruzamento é chamado de *validação externa* ou *validação a posteriori*: verificamos se a estrutura descoberta pelo algoritmo corresponde a uma classificação oficial que ele nunca viu. A palavra "temporal" indica que o objetivo final é comparar os clusters de 2010 com os de 2022 para identificar trajetórias de mudança, não avaliar desempenho de um modelo preditivo.

### 6.1 Clustering do CNEFE 2010

Aplicamos HDBSCAN diretamente nos 12 features z-score do CNEFE 2010, **sem UMAP**. Isso é diferente do pipeline de 2022, onde o UMAP reduziu os dados para 2D antes do HDBSCAN.

```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=200,   # mais conservador: ver justificativa abaixo
    min_samples=3,
    metric='euclidean',
    cluster_selection_method='eom'
)
df_2010['cluster'] = clusterer.fit_predict(X)
```

**Por que `min_cluster_size=200` para 2010 e `50` para 2022?**

A diferença não é arbitrária — ela decorre de uma diferença estrutural entre os dois pipelines:

No pipeline de 2022, o UMAP comprime os dados de 11D para 2D antes do HDBSCAN. Nesse espaço 2D, setores similares ficam fisicamente próximos e formam ilhas compactas visíveis. Com a densidade já concentrada pela redução dimensional, 50 pontos são suficientes para o HDBSCAN identificar uma região densa como cluster.

No pipeline de 2010, o HDBSCAN opera diretamente em 12D, sem UMAP. Em alta dimensão, a estimativa de densidade é menos confiável: os pontos estão mais espalhados no espaço e pequenas flutuações locais podem ser confundidas com estrutura real. Para que o HDBSCAN só reconheça como cluster algo genuinamente denso e não ruído dimensional, o limiar mínimo precisa ser maior. Com `min_cluster_size=50` em 12D, o algoritmo fragmentaria setores semanticamente idênticos em dezenas de micro-clusters por conta de variações locais nas proporções de logradouro.

Um segundo fator é a natureza das features de 2010: os tipos de logradouro têm distribuição mais binária do que o COD_ESPECIE de 2022. Muitos setores têm 90-100% de uma só categoria (`prop_rua` ou `prop_fazenda`), o que cria regiões densíssimas em cantos específicos do espaço de features, separadas por vazios. Sem UMAP para redistribuir essa estrutura, o HDBSCAN com limiar pequeno superFragmentaria essas regiões densas em clusters geograficamente distintos mas funcionalmente idênticos.

![Perfil z-score dos clusters CNEFE 2010 BA](outputs/figures/perfil_clusters_2010_ba.png)

*Heatmap de z-scores para os clusters do CNEFE 2010. As features aqui são proporções de tipo de logradouro (`prop_rua`, `prop_fazenda`, etc.), não COD_ESPECIE. Os clusters rurais se destacam em `prop_fazenda`, `prop_estrada` e `prop_sitio`; os urbanos, em `prop_rua` e `prop_avenida`. A separação visual confirma que mesmo com features mais simples de 2010 o algoritmo captura a estrutura urbano/rural.*

![Single Linkage Tree: CNEFE 2010 BA](outputs/figures/single_linkage_tree_2010_ba.png)

*Árvore de condensação do HDBSCAN para 2010. Com `min_cluster_size=200` (vs 50 em 2022), a árvore é mais conservadora e produz menos clusters. O galho mais persistente e largo corresponde ao grande cluster de setores urbanos residenciais; galhos mais finos e curtos correspondem a grupos rurais específicos.*

### 6.2 Validação a Posteriori: 92% de Pureza

Com os clusters definidos sem usar `situacao`, cruzamos o resultado com o campo oficial. `pd.crosstab` gera uma matriz cluster × situacao_oficial com contagens: cada linha é um cluster, cada coluna é uma categoria oficial. `idxmax(axis=1)` pega o nome da coluna com maior contagem em cada linha (a categoria dominante); `max(axis=1) / sum(axis=1)` divide esse máximo pelo total da linha para obter a fração. A pureza final é ponderada pelo tamanho de cada cluster para não dar peso igual a clusters com 50 e 5.000 setores:

```python
df_val['situacao_oficial'] = (df_val['prop_urbano'] > 0.5).map({True: 'Urbano', False: 'Rural'})

ct        = pd.crosstab(df_val['cluster'], df_val['situacao_oficial'])
dominante = ct.idxmax(axis=1)          # categoria mais frequente em cada cluster
pureza    = ct.max(axis=1) / ct.sum(axis=1)  # fração do grupo dominante
n         = ct.sum(axis=1)             # tamanho de cada cluster

pureza_media_ponderada = (pureza * n).sum() / n.sum()  # média ponderada pelo tamanho
# → 92%
```

**92% de pureza ponderada** significa que 92% dos setores de cada cluster pertencem à mesma categoria oficial (todos urbanos ou todos rurais). O algoritmo recuperou a estrutura urbano/rural usando apenas o tipo de logradouro confirmando que as features capturam algo real.

### 6.3 Experimento COM vs SEM Coordenadas Geográficas

Para verificar se adicionar coordenadas melhora o clustering, realizamos um experimento controlado. O CNEFE 2010 não tem coordenadas nativas  usamos os centroides dos setores do Censo 2022 como proxy geográfico (limitação declarada explicitamente).

**Ponto metodológico central:** para que os Silhouette Scores sejam comparáveis, ambas as configurações são avaliadas **no mesmo espaço** (14 features de proporção de tipo), usando o mesmo `StandardScaler`:

```python
# SEM geo: avaliado em 14D
score_sem = silhouette_score(X[mask_sem], df_2010['cluster'].values[mask_sem])

# COM geo: também avaliado em 14D, usa scaler.transform (não fit_transform)
# para manter a mesma escala e tornar os scores comparáveis
X_prop_geo = scaler.transform(df_geo[FEATURES])
score_com  = silhouette_score(X_prop_geo[mask_com], labels_geo[mask_com])
```

Usar `scaler.transform()` em vez de `fit_transform()` é essencial: recalibrar o scaler no subconjunto COM geo produziria z-scores em escala diferente, tornando a comparação inválida.

![Comparativo SEM vs COM coordenadas geográficas: CNEFE 2010 BA](outputs/figures/comparativo_sem_com_geo_2010_ba.png)

*Comparação lado a lado dos clusters obtidos com e sem coordenadas geográficas como features adicionais. A adição de latitude/longitude como features tende a fragmentar clusters funcionalmente homogêneos em subgrupos por localização geográfica, aumentando o número de clusters mas reduzindo a pureza semântica. O gráfico mostra os Silhouette Scores avaliados no mesmo espaço de proporções (14 features), garantindo comparação justa entre as duas configurações.*

---

## 7. Análise Temporal 2010 → 2022

### 7.1 O Desafio do Join

Para cruzar os dois Censos, é preciso unir os DataFrames pelo código do setor censitário. O problema: o IBGE adotou convenções diferentes em cada edição.

- **2010:** `293330705000177` (15 dígitos, sem sufixo)
- **2022:** `291840705000165P` (15 dígitos + `'P'` no final)

O sufixo `'P'` indica que o registro é do tipo *ponto de endereço* no CNEFE 2022 uma distinção interna do IBGE que não existia em 2010. Como os campos são strings, `'291840705000165P' != '291840705000165'` e o join retorna 0 matches sem tratamento.

A correção é remover o último caractere de todos os códigos de 2022 com `str[:-1]` (fatia de string que descarta a posição `-1`, ou seja, o último caractere):

```python
# Antes:  '291840705000165P'
# Depois: '291840705000165'
df_2022['cod_setor'] = df_2022['cod_setor'].str[:-1]

df = df_2022.merge(
    df_2010[['cod_setor', 'total_2010', 'prop_urbano_2010', 'prop_rural_2010']],
    on='cod_setor', how='left'
)
```

Após a correção: **19.371 setores presentes nos dois anos**, base da análise temporal. Os demais 10.984 ficam com `total_2010 = NaN` e são classificados como "Novo Setor (2022)".

### 7.2 Métricas de Mudança

Para cada setor presente nos dois anos calculamos dois indicadores. `delta_abs` é a variação bruta de endereços (positivo = cresceu, negativo = encolheu). `taxa_cresc` normaliza pelo tamanho original do setor em 2010  sem isso, um setor pequeno que dobrou de tamanho e um grande que cresceu pouco teriam pesos equivalentes no `delta_abs`, o que distorceria a análise:

```python
df_match['delta_abs']  = df_match['total_2022'] - df_match['total_2010']
df_match['taxa_cresc'] = (df_match['total_2022'] - df_match['total_2010']) / df_match['total_2010']
```

Crescimento médio entre 2010 e 2022: **+24,8% de endereços por setor** (mediana +14,1%).

### 7.3 Classificação de Trajetórias

A função usa dois eixos de informação para cada setor: a composição em 2010 (o que ele era) e a taxa de crescimento até 2022 (o que aconteceu com ele). As regras são avaliadas em ordem  a primeira que for verdadeira determina o rótulo. Os limiares foram calibrados para capturar mudanças estruturais: 0.3 (30%) de crescimento separa expansão real de variação normal de recenseamento, e 0.5 de `prop_rural` identifica setores predominantemente rurais, não apenas com alguma presença rural:

```python
def classificar_trajetoria(row):
    if row['prop_rural_2010'] > 0.5 and row['taxa_cresc'] > 0.3:
        return 'Rural → Urbano'       # era rural e cresceu >30%: urbanização
    if row['prop_urbano_2010'] > 0.8 and row['taxa_cresc'] > 0.3:
        return 'Adensamento Urbano'   # já era urbano e adensou ainda mais
    if row['taxa_cresc'] < -0.2:
        return 'Declínio'             # perdeu >20% dos endereços
    if abs(row['taxa_cresc']) <= 0.2:
        return 'Estável'              # variação dentro de ±20%
    return 'Crescimento Moderado'     # cresceu entre 20% e 30%
```

| Trajetória                 | Setores |
| --------------------------- | ------- |
| Novo Setor (apenas em 2022) | 10.984  |
| Estável                    | 8.118   |
| Rural → Urbano             | 3.883   |
| Adensamento Urbano          | 3.487   |
| Crescimento Moderado        | 2.533   |
| Declínio                   | ~756    |

**10.984 setores novos em 2022** quase 1 em cada 3 setores da BA não existia em 2010. Isso representa expansão territorial real, não apenas revisão de limites.

O cruzamento **trajetória × cluster 2022** revela que o tipo de setor prediz a trajetória: setores classificados como "Residencial Denso" em 2022 concentram os casos de adensamento urbano; setores "Rural Agrícola" tendem a permanecer estáveis ou declinar.

![Perfil de clustering 2010 na Bahia](outputs/figures/mapa_perfil_2010_ba.png)

*Distribuição geográfica dos clusters do CNEFE 2010 na Bahia. Como o CNEFE 2010 não tem coordenadas por endereço, os setores são plotados pelo centroide aproximado. A separação litoral/sertão é visível mesmo com features apenas de tipo de logradouro.*

![Perfil de clustering 2022 na Bahia](outputs/figures/mapa_perfil_2022_ba.png)

*Mesma visualização para 2022, agora com coordenadas precisas por endereço. Permite comparar visualmente a evolução da estrutura de uso do solo entre os dois censos: Salvador e Feira de Santana ganham setores Residenciais Densos; o interior mantém o perfil Rural.*

![Transições de perfil entre 2010 e 2022](outputs/figures/mapa_transicoes_2010_2022_ba.png)

*Mapa colorido pela mudança de perfil de cada setor entre 2010 e 2022. Setores que transitaram de Rural para Residencial aparecem em uma cor de transição; setores estáveis mantêm sua cor original. As franjas de expansão urbana ao redor das principais cidades são visualmente identificáveis.*

![Matriz de transição de clusters 2010 para 2022](outputs/figures/matriz_transicao_2010_2022_ba.png)

*Matriz de contingência mostrando quantos setores de cada cluster de 2010 migraram para cada cluster de 2022. A diagonal principal representa setores que mantiveram o mesmo perfil; fora da diagonal, mudanças de tipo. Clusters rurais de 2010 com fluxo significativo para clusters Residenciais em 2022 indicam as áreas de urbanização mais intensa.*

![Taxa de crescimento de endereços por cluster (2010 a 2022)](outputs/figures/crescimento_por_cluster_ba.png)

*Crescimento percentual médio de endereços por cluster entre 2010 e 2022. Clusters Residenciais Densos e Em Construção tendem a mostrar as maiores taxas, confirmando que o tipo de setor em 2022 é preditor da dinâmica de crescimento. Clusters Rurais Agrícolas apresentam crescimento próximo de zero ou negativo.*

![Trajetória dos setores 2010 a 2022 no mapa da Bahia](outputs/figures/mapa_trajetoria_temporal_ba.png)

*Cada setor colorido pela trajetória classificada pela função `classificar_trajetoria`. Setores "Rural → Urbano" (em laranja ou vermelho) concentram-se na periferia das cidades médias. "Adensamento Urbano" aparece nos anéis intermediários das capitais regionais. "Declínio" é visível em municípios do sertão com êxodo rural.*

![Cruzamento trajetória temporal por cluster 2022](outputs/figures/trajetoria_vs_cluster_ba.png)

*Matriz de frequência cruzando trajetória (2010→2022) com o cluster atribuído em 2022. Confirma que o tipo de setor em 2022 tem poder preditivo sobre a trajetória: "Residencial Denso" concentra "Adensamento Urbano"; "Rural Agrícola" concentra "Estável" e "Declínio"; clusters Em Construção têm alta proporção de setores "Novos".*

---

## 8. Comparação de Algoritmos: DBSCAN vs HDBSCAN

Um ponto metodológico essencial antes de qualquer comparação: **tanto o DBSCAN quanto o HDBSCAN recebem como entrada as mesmas coordenadas 2D geradas pelo UMAP** (`umap_x`, `umap_y`). O UMAP não é uma vantagem exclusiva do HDBSCAN é uma etapa de pré-processamento compartilhada. O que varia entre os algoritmos é apenas como cada um particiona esse mesmo espaço 2D. Isso torna a comparação direta e justa: qualquer diferença nos resultados vem do comportamento do algoritmo, não da entrada.

```
11 features (z-score) → UMAP (11D → 2D) → { DBSCAN ε=0.30  → 19 clusters, 1.1% ruído }
                                            { DBSCAN ε=1.50  → 13 clusters, 0.3% ruído }
                                            { HDBSCAN        → 13 clusters, 0.3% ruído }
```

### 8.1 O Problema do Parâmetro ε no DBSCAN

O DBSCAN exige a escolha manual do parâmetro ε (raio de vizinhança). A ferramenta padrão para isso é o **k-distance plot**: ordena os pontos pela distância ao k-ésimo vizinho e procura um "cotovelo" que indique a densidade natural dos dados. O código abaixo calcula essa distância para cada ponto com `NearestNeighbors`, pega a distância ao 3º vizinho (`dists[:, k-1]`, índice `k-1` pois é base-zero), e ordena do maior para o menor (`[::-1]`) para revelar o cotovelo, se existir:

```python
from sklearn.neighbors import NearestNeighbors

k = 3
nbrs = NearestNeighbors(n_neighbors=k).fit(X_umap)
dists, _ = nbrs.kneighbors(X_umap)
kdist = np.sort(dists[:, k-1])[::-1]   # distância ao 3º vizinho, ordem decrescente
```

No caso dos setores censitários da Bahia, o k-distance plot **não mostra cotovelo claro**. A curva declina de forma suave e contínua  os dados têm densidades muito diferentes entre os grupos (setores urbanos de Salvador são muito mais densos que setores rurais do sertão). Qualquer ε escolhido será arbitrário.

![K-distance plot para os setores censitários da BA](outputs/figures/kdistance_plot.png)

*O k-distance plot ordena todos os setores pela distância ao 3º vizinho mais próximo no espaço UMAP (eixo y), do maior para o menor (eixo x). Um "cotovelo" claro indicaria o ε ideal. A ausência de cotovelo aqui, com a curva declinando suavemente, é o diagnóstico visual de que os dados têm múltiplas densidades sobrepostas: não existe um único valor de ε que funcione bem para todos os grupos simultaneamente.*

### 8.2 Varredura de ε: O Dilema do Dom. Coletivo

| ε                | Clusters     | Ruído %       | Dom. Coletivo?           |
| ----------------- | ------------ | -------------- | ------------------------ |
| 0.05              | 898          | 21.1%          | ✓                       |
| 0.10              | 203          | 7.5%           | ✓                       |
| 0.30              | 19           | 1.1%           | ✓                       |
| 1.00              | 14           | 0.4%           | ✓                       |
| **1.50**    | **13** | **0.3%** | **✗ PERDIDO**     |
| **HDBSCAN** | **13** | **0.3%** | **✓ automático** |

Com ε=1.50, o DBSCAN chega ao mesmo número de clusters que o HDBSCAN e ao mesmo nível de ruído, mas **perde os 175 setores de Domicílio Coletivo** (hospitais, presídios, asilos), que são absorvidos pelo cluster Residencial. Não é um detalhe estatístico: o algoritmo passa a classificar um hospital da mesma forma que uma rua residencial.

![Comparação DBSCAN ε=0.30 no espaço UMAP](outputs/figures/comp_dbscan_030.png)

*DBSCAN com ε=0.30 no espaço UMAP. Com raio pequeno, o algoritmo é sensível à estrutura local e detecta 19 clusters, incluindo Dom. Coletivo. O lado negativo é 1.1% de ruído e fragmentação excessiva: tipos funcionalmente idênticos ficam em clusters separados apenas por pequenas diferenças de densidade local.*

![Comparação DBSCAN ε=1.50 no espaço UMAP](outputs/figures/comp_dbscan_150.png)

*DBSCAN com ε=1.50 no espaço UMAP. Com raio maior, o algoritmo "cola" regiões de densidade variável e reduz para 13 clusters com apenas 0.3% de ruído, mesmos números do HDBSCAN. O problema: o raio agora é grande o suficiente para absorver os 175 setores Dom. Coletivo (cluster compacto mas pequeno) dentro do cluster Residencial adjacente. O cluster Dom. Coletivo desaparece do resultado.*

![Comparação HDBSCAN no espaço UMAP](outputs/figures/comp_hdbscan.png)

*HDBSCAN com min_cluster_size=50 no mesmo espaço UMAP. Obtém 13 clusters e 0.3% de ruído como o DBSCAN ε=1.50, mas mantém o cluster Dom. Coletivo intacto. A diferença é estrutural: o HDBSCAN avalia densidade localmente em múltiplas escalas simultaneamente, permitindo que regiões esparsas (Rural) e regiões compactas (Dom. Coletivo) coexistam como clusters válidos.*

### 8.3 Por que a Densidade Variável é o Problema Fundamental

O DBSCAN trata densidade como uniforme: define um único ε para todo o espaço. O HDBSCAN é **hierárquico**: constrói uma árvore de dendrograma e seleciona clusters em diferentes níveis de densidade, preservando tanto os grupos compactos (Dom. Coletivo 175 setores muito específicos) quanto os grupos difusos (Rural muitos setores com composição variada). Ao contrário do ε do DBSCAN que não tem unidade intuitiva no espaço UMAP , `min_cluster_size` tem interpretação direta: "só considere um grupo como cluster se ele contiver ao menos 50 setores":

```python
# HDBSCAN: sem ε, parâmetro com interpretação direta
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,   # threshold mínimo de tamanho para existir como cluster
    min_samples=3,
    cluster_selection_method='eom'
)
```

### 8.4 Resumo da Comparação

| Critério                        | DBSCAN ε=0.30 | DBSCAN ε=1.50 | HDBSCAN                         |
| -------------------------------- | -------------- | -------------- | ------------------------------- |
| Clusters                         | 19             | 13             | 13                              |
| Ruído %                         | 1.1%           | 0.3%           | 0.3%                            |
| Dom. Coletivo preservado?        | ✓             | **✗**   | ✓                              |
| Parâmetro principal             | ε arbitrário | ε arbitrário | min_cluster_size interpretável |
| Robustez a densidades variáveis | Baixa          | Baixa          | Alta                            |

Os mapas semânticos abaixo mostram a distribuição geográfica dos tipos de setor para cada configuração, evidenciando como o HDBSCAN mantém o cluster Dom. Coletivo (vermelho) enquanto o DBSCAN com ε=1.50 o dissolve no cluster Residencial:

![Mapa semantico DBSCAN ε=0.30](outputs/figures/mapa_semantico_dbscan_030.png)

*Mapa geográfico com os setores coloridos pelo tipo semântico atribuído pelo DBSCAN ε=0.30. Com 19 clusters, há mais tipos distintos no mapa, mas a legenda fica mais fragmentada. O Dom. Coletivo (vermelho) é visível nas cidades polo. A fragmentação excessiva dificulta a leitura regional.*

![Mapa semantico DBSCAN ε=1.50](outputs/figures/mapa_semantico_dbscan_150.png)

*Mesmo mapa para DBSCAN ε=1.50. O Dom. Coletivo (vermelho) desaparece: os hospitais, presídios e asilos da Bahia agora são classificados como Residencial (azul), tornando impossível qualquer análise que dependa da identificação dessas instalações. O mapa parece mais "limpo", mas esconde informação crítica.*

![Mapa semantico HDBSCAN](outputs/figures/mapa_semantico_hdbscan.png)

*Mapa para o HDBSCAN. 13 clusters, 0.3% de ruído e Dom. Coletivo preservado (pontos vermelhos visíveis em Salvador e nas cidades regionais). Este é o resultado final adotado pelo trabalho: a melhor combinação de cobertura, granularidade e preservação de grupos minoritários importantes para análise de saúde pública.*

---

## 9. Resultados e Interpretação

### 9.1 Os 13 Clusters do Censo 2022

| Cluster | Nome Semântico                     | n      | Feature dominante                                                          |
| ------- | ----------------------------------- | ------ | -------------------------------------------------------------------------- |
| 0–3    | Residencial Puro (núcleos A–D)    | ~6.000 | `prop_domicilio_particular` alto, sem espécie secundária significativa |
| 4       | Residencial Puro                    | ~3.500 | Mesmo perfil dos anteriores, distribuição geográfica mais homogênea    |
| 5       | Residencial Denso                   | ~2.800 | Dom. particular + finalidade residencial muito altos, alta densidade       |
| 6       | Rural / Peri-urbano Misto           | ~1.200 | Agropecuário + estab. outras, baixa densidade                             |
| 7       | Residencial c/ Educacional          | ~900   | Dom. particular + pico em`prop_estab_ensino`                             |
| 8       | Uso Misto Comercial                 | ~800   | `prop_estab_outras` e `prop_finalidade_comercial` elevados             |
| 9       | Rural (baixa densidade)             | ~600   | `prop_estab_agropecuario` dominante                                      |
| 10      | Rural Agrícola                     | ~400   | `prop_estab_agropecuario` muito alto, finalidade mista                   |
| 11      | Domicílio Coletivo (Institucional) | ~175   | `prop_domicilio_coletivo` extremamente alto                              |
| 12      | Setor Urbano Típico                | ~7.500 | Perfil médio urbano equilibrado                                           |

### 9.2 Distribuição Geográfica

A distribuição no mapa da Bahia confirma coerência geográfica:

- **Litoral e Salvador:** Residencial Puro, Residencial Denso, Setor Urbano Típico
- **Interior e sertão:** Rural Agrícola, Rural (baixa densidade)
- **Dom. Coletivo:** concentrado nas cidades polo  Salvador, Feira de Santana, Vitória da Conquista, Ilhéus

### 9.3 Impacto Prático

**Saúde pública:** os 175 setores Dom. Coletivo (hospitais, presídios, asilos) são automaticamente identificados sem consulta manual a cadastros. O DBSCAN com ε=1.50 perde essa informação, tornando impossível qualquer análise de cobertura de saúde em domicílios coletivos.

**Planejamento urbano:** os 10.984 setores novos em 2022, cruzados com a trajetória Rural→Urbano, indicam fronteiras de expansão  áreas prioritárias para infraestrutura de saneamento, transporte e saúde.

**Pesquisa:** comparação temporal padronizada entre censos com schemas incompatíveis, possível apenas via abordagem não supervisionada sobre a estrutura intrínseca dos dados.

---

## 10. Aderência aos Requisitos do Projeto

### 10.1 Execução do Pré-processamento

O pré-processamento é executado inteiramente no notebook `04_preprocessamento.ipynb` e compreende quatro etapas encadeadas: (1) leitura dos arquivos Parquet via DuckDB sem carregamento em memória, (2) filtragem de endereços sem geocodificação e de setores com menos de 5 endereços, (3) agregação de 9 milhões de linhas em 30.355 vetores de proporção por setor via SQL, e (4) normalização z-score com `StandardScaler`. O produto final, `outputs/setores_features.parquet`, é o único artefato que os notebooks de modelagem consomem — os dados brutos do IBGE não são acessados novamente após essa etapa.

Para o CNEFE 2010, o mesmo processo é executado em `08_clustering_2010_ba.ipynb`: leitura dos arquivos `.snappy.parquet` particionados, cálculo de proporções de 12 tipos de logradouro por setor, normalização e clustering.

### 10.2 Aplicação dos Modelos

Dois algoritmos são aplicados e comparados sistematicamente:

**UMAP + HDBSCAN** (`05_modelagem.ipynb`): pipeline principal para o Censo 2022. O UMAP reduz 11 features z-score para 2 dimensões preservando estrutura topológica local. O HDBSCAN aplicado no espaço 2D retorna 13 clusters com 0,3% de ruído. O mesmo algoritmo é aplicado ao CNEFE 2010 em `08_clustering_2010_ba.ipynb` com `min_cluster_size=200`, retornando 11 clusters.

**DBSCAN** (`09_comparacao_algoritmos.ipynb`): aplicado com varredura de ε (0.05, 0.10, 0.30, 1.00, 1.50) sobre as mesmas projeções UMAP, servindo como baseline de comparação. O experimento documenta quantitativamente por que o HDBSCAN é superior neste conjunto de dados.

### 10.3 Validação dos Resultados

A validação ocorre em três camadas independentes:

**Validação interna** (métricas de clustering): Silhouette Score calculado no espaço UMAP (0,31) e no espaço 11D original (-0,10), com discussão explícita de por que a discrepância é esperada e não invalida os clusters. K-distance plot como diagnóstico visual da ausência de estrutura uniforme de densidade.

**Validação externa a posteriori** (`08_clustering_2010_ba.ipynb`): os clusters do CNEFE 2010 são cruzados com o campo `situacao` (urbano/rural), que não foi usado em nenhum momento como feature. A pureza ponderada de **92%** confirma que o algoritmo captura estrutura real nos dados sem supervisão.

**Validação temporal** (`07_analise_temporal_ba.ipynb`): os clusters de 2022 são cruzados com as trajetórias 2010→2022. O fato de que o cluster "Residencial Denso" concentra casos de "Adensamento Urbano" e que o cluster "Rural Agrícola" concentra "Declínio" confirma coerência semântica dos tipos descobertos ao longo do tempo.

### 10.4 Criatividade na Escolha do Problema

O problema tem dois elementos de originalidade que vão além da aplicação direta de algoritmos conhecidos.

O primeiro é a **incompatibilidade de schemas como motivação metodológica**: os dois censos do IBGE (2010 e 2022) descrevem o mesmo fenômeno com campos completamente diferentes, tornando qualquer comparação temporal direta impossível. A solução não supervisionada não é um contorno técnico — é a única abordagem que permite descobrir estrutura comparável nos dois schemas sem suposições de mapeamento arbitrário.

O segundo é a **descoberta de tipologias não oficiais com validação cruzada no tempo**: o IBGE não publica tipologias funcionais de setor censitário além do binário urbano/rural. Os 13 tipos descobertos (Domicílio Coletivo, Rural Agrícola, Residencial Denso, Uso Misto Comercial, etc.) são um produto original que não existe em nenhum cadastro oficial, com aplicações diretas em planejamento urbano e saúde pública. A trajetória de 3.883 setores classificados como "Rural → Urbano" entre os dois censos é um resultado que o IBGE não publica e que só emerge da combinação de clustering não supervisionado com análise temporal.

---

## 11. Estrutura do Repositório

```
.
├── data/
│   ├── cnefe_2010/          # CNEFE 2010 particionado (.snappy.parquet)
│   └── censo 2022/          # Censo 2022 BA (29_BA.parquet + dicionário .xls)
│
├── notebooks/
│   ├── 01_eda_cnefe.ipynb              # EDA: CNEFE 2010
│   ├── 02_eda_web_scraping.ipynb       # EDA: dados auxiliares
│   ├── 03_eda_censo_2022.ipynb         # EDA: Censo 2022 (schema, nulos, distribuições)
│   ├── 04_preprocessamento.ipynb       # Engenharia de features (DuckDB → Parquet)
│   ├── 05_modelagem.ipynb              # Pipeline UMAP → HDBSCAN (2022)
│   ├── 07_analise_temporal_ba.ipynb    # Análise temporal 2010→2022 (trajetórias)
│   ├── 08_clustering_2010_ba.ipynb     # Clustering CNEFE 2010 + validação + experimento geo
│   └── 09_comparacao_algoritmos.ipynb  # DBSCAN vs HDBSCAN (k-distance, varredura ε, mapas)
│
├── outputs/
│   ├── setores_features.parquet            # 30.355 setores × 11 features (z-scores)
│   ├── setores_clusterizados.parquet       # + cluster, cluster_nome, cluster_tipo, umap_x/y
│   ├── setores_clusterizados_2010_ba.parquet
│   ├── analise_temporal_ba.parquet         # Trajetórias 2010→2022
│   └── figures/                            # Todos os mapas e gráficos gerados
│
├── requirements.txt
└── README.md
```

### Dependências e Reprodução

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Executar na ordem
jupyter nbconvert --to notebook --execute notebooks/04_preprocessamento.ipynb --output-dir notebooks/
jupyter nbconvert --to notebook --execute notebooks/05_modelagem.ipynb --output-dir notebooks/
jupyter nbconvert --to notebook --execute notebooks/07_analise_temporal_ba.ipynb --output-dir notebooks/
jupyter nbconvert --to notebook --execute notebooks/08_clustering_2010_ba.ipynb --output-dir notebooks/
jupyter nbconvert --to notebook --execute notebooks/09_comparacao_algoritmos.ipynb --output-dir notebooks/
```

---

## 12. Perguntas Frequentes sobre a Metodologia

**O IBGE já classifica cada endereço por `COD_ESPECIE`. O que este trabalho acrescenta que o IBGE não entrega?**

O IBGE classifica cada *endereço* individualmente: "este endereço é um domicílio particular", "este é um estabelecimento de saúde". O que o IBGE não entrega é uma *tipologia de setor censitário* baseada na composição funcional agregada. O produto oficial é binário: urbano ou rural. As tipologias descobertas aqui (Residencial Denso, Rural Agrícola, Domicílio Coletivo (Institucional), Uso Misto Comercial) não existem em nenhuma publicação do IBGE. Usamos a classificação por endereço como matéria-prima; o que geramos como saída é distinto e mais granular do que qualquer produto oficial disponível.

---

**Por que proporções em vez de contagens absolutas de endereços?**

Contagens absolutas acoplam a composição funcional ao tamanho do setor. Um setor residencial com 800 domicílios e um setor residencial com 80 ficam numericamente distantes mesmo tendo o mesmo perfil de uso do solo. Com proporções, dois setores com composição 90% residencial ficam próximos no espaço de features independentemente do tamanho, que é o objetivo: comparar *perfil*, não *volume*. O volume (`total_enderecos`) é mantido como atributo descritivo separado e usado nas análises temporais de crescimento, mas nunca como feature de clustering.

---

**Por que aprendizado não supervisionado se existe o campo `situacao` (urbano/rural) no CNEFE 2010?**

Duas razões independentes, cada uma suficiente por si só.

Primeira: o campo `situacao` existe apenas no CNEFE 2010; o Censo 2022 não tem campo equivalente. Uma abordagem supervisionada treinada em 2010 não pode ser aplicada em 2022 sem uma cadeia de mapeamentos arbitrários entre os dois schemas.

Segunda: o objetivo não é replicar a classificação urbano/rural, mas descobrir subtipos dentro de cada classe. Um modelo supervisionado com `situacao` como alvo aprenderia a separar urbano de rural e nada mais. As tipologias que buscamos (hospitais vs. residências vs. fazendas) existem dentro do grupo "urbano" e dentro do grupo "rural" simultaneamente. Usar `situacao` como feature tornaria a análise circular: encontraríamos clusters de "urbano" e "rural" porque incluímos exatamente esses rótulos na entrada.

O cruzamento com `situacao` acontece apenas *depois* do clustering, como validação: o fato de 92% de pureza ser atingida sem nunca ter visto o campo confirma que a abordagem captura estrutura real.

---

**Por que HDBSCAN e não DBSCAN?**

O DBSCAN exige um único parâmetro ε (raio de vizinhança) aplicado uniformemente a todos os pontos. Nos setores censitários da Bahia, as densidades são muito heterogêneas: Salvador tem setores urbanos extremamente compactos; o sertão tem setores rurais espalhados. O k-distance plot sem cotovelo (seção 8.1) é o diagnóstico quantitativo desse problema.

A consequência prática está na varredura de ε (seção 8.2): qualquer valor que controle bem o ruído nos setores urbanos é grande demais para os rurais, e o único ponto em que o DBSCAN atinge 0.3% de ruído (ε=1.50) é justamente onde perde os 175 setores de Domicílio Coletivo (hospitais, presídios e asilos) que constituem um grupo funcionalmente distinto e relevante para análise de saúde pública.

O HDBSCAN resolve isso avaliando densidade localmente em múltiplas escalas via árvore hierárquica, sem exigir que todas as regiões do espaço tenham a mesma densidade. O resultado com `min_cluster_size=50` atinge os mesmos 13 clusters e 0.3% de ruído que o DBSCAN ε=1.50, mantendo o cluster Dom. Coletivo intacto.

---

**Por que Parquet e DuckDB em vez de CSV e Pandas?**

Com ~9 milhões de linhas apenas para a Bahia, o CSV do CNEFE 2022 ocupa mais de 2 GB em disco e esgotaria a RAM de uma máquina de desenvolvimento comum ao ser carregado via `pd.read_csv()`. Parquet resolve o armazenamento (colunar + compressão Snappy, ~8x menor que CSV) e DuckDB resolve a agregação: executa a query SQL diretamente sobre o arquivo Parquet sem carregar o conjunto completo em memória, reduzindo 9 milhões de linhas para ~30 mil setores em segundos. A escolha não é preferência técnica, mas necessidade prática para o volume dos dados.

---

**O Silhouette Score negativo em 11D não indica que os clusters são ruins?**

O Silhouette Score mede separação euclidiana entre clusters. O UMAP com `min_dist=0.0` forma clusters baseados em *vizinhança topológica*: dois setores pertencem ao mesmo cluster se estão conectados por uma cadeia de vizinhos próximos, mesmo que a distância euclidiana direta em 11D entre eles seja grande. Isso é esperado e deliberado: setores em cidades diferentes da Bahia têm o mesmo perfil funcional (mesmas proporções de COD_ESPECIE) mas foram separados geograficamente pelo UMAP, criando clusters topologicamente coesos que parecem sobrepostos quando medidos por distância euclidiana em 11D.

O Silhouette positivo no espaço UMAP (0.31) confirma que os clusters fazem sentido no espaço onde foram criados. O negativo em 11D não invalida o resultado: é uma consequência previsível e documentada de usar uma métrica euclidiana para avaliar clusters topológicos.

---

**Por que `min_cluster_size=50` para 2022 e `200` para 2010?**

A diferença vem de uma assimetria estrutural entre os dois pipelines. Para 2022, o HDBSCAN recebe como entrada o espaço 2D do UMAP, onde setores similares já estão fisicamente compactos. Nesse espaço comprimido, 50 pontos formam uma região densa suficientemente estável para o algoritmo reconhecer como cluster.

Para 2010, o HDBSCAN opera diretamente em 12D, sem UMAP. Em alta dimensão, a estimativa de densidade é inerentemente menos confiável: pequenas variações locais nas proporções de logradouro podem gerar falsas densidades. Um `min_cluster_size=50` em 12D causaria fragmentação excessiva, separando em clusters distintos setores que são funcionalmente o mesmo tipo apenas porque estão em regiões geográficas diferentes e têm pequenas diferenças nas proporções de `prop_rua` ou `prop_fazenda`. O valor de 200 exige que o HDBSCAN só reconheça como cluster algo com densidade suficiente para sobreviver ao ruído dimensional.

Em proporção do total de setores: 50/30.355 ≈ 0,16% (2022 em 2D) vs 200/23.763 ≈ 0,84% (2010 em 12D). Ambos são conservadores, mas o segundo compensa a ausência da redução dimensional.

---

## 13. Referências Bibliográficas

### Algoritmos e Métodos

**[1]** Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In *Advances in Knowledge Discovery and Data Mining PAKDD 2013*, Lecture Notes in Computer Science, vol. 7819, pp. 160–172. Springer, Berlin, Heidelberg.

> Artigo original do HDBSCAN. Define o conceito de *mutual reachability distance* e *condensed cluster tree*, que permitem identificar clusters de densidades variáveis sem a necessidade de um parâmetro ε fixo, motivação central para a escolha do algoritmo neste trabalho.

**[2]** McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.

> Artigo original do UMAP. Demonstra que a projeção em baixa dimensão preserva estrutura topológica local e global. Justifica o uso de `min_dist=0.0` para maximizar a separação entre grupos e viabilizar o HDBSCAN downstream.

**[3]** Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)*, pp. 226–231. AAAI Press.

> Artigo original do DBSCAN. Introduz o conceito de *core points* e o parâmetro ε. Usado como baseline de comparação no notebook 09  suas limitações com densidades variáveis (documentadas no k-distance plot sem cotovelo claro) motivam a adoção do HDBSCAN.

**[4]** Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.

> Define o Silhouette Score como métrica de coesão e separação interna de clusters. Utilizado para comparação entre clustering em 11D e em espaço UMAP, com discussão explícita de por que o score pode ser negativo quando os clusters são topológicos e não esféricos.

**[5]** Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? In *Proceedings of the 7th International Conference on Database Theory (ICDT 1999)*, Lecture Notes in Computer Science, vol. 1540, pp. 217–235. Springer.

> Formaliza o problema da maldição da dimensionalidade para buscas por vizinhança: em alta dimensão, a razão entre distância máxima e mínima ao vizinho mais próximo converge para 1, tornando o conceito de "proximidade" degenerado. Fundamenta a necessidade do UMAP antes do HDBSCAN.

### Contexto e Estado da Arte

**[6]** Singleton, A. D., & Longley, P. A. (2009). Geodemographics, visualisation, and social distinctions in urban areas. *Social & Cultural Geography*, 10(6), 727–750.

> Discute o uso de clustering não supervisionado sobre dados censitários para criar classificações geodemográficas de áreas urbanas. Demonstra que tipologias descobertas por clustering superam a granularidade das classificações oficiais para análise de políticas públicas  abordagem metodologicamente equivalente à deste trabalho.

**[7]** Venerandi, A., Quattrone, G., & Capra, L. (2018). A scalable method to quantify the relationship between urban form and socioeconomic indexes. In *Proceedings of the 26th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, pp. 408–411.

> Aplica representação vetorial de composição de uso do solo para caracterizar setores urbanos, similar ao vetor de proporções deste trabalho. Suporte direto para a escolha de features baseadas em proporções de tipo de estabelecimento em vez de contagens absolutas.

**[8]** Fonseca, F., & Bação, F. (2005). Classificação de áreas urbanas: uma abordagem baseada em sistemas de auto-organização. *Finisterra Revista Portuguesa de Geografia*, 40(79), 61–78.

> Aplica mapas auto-organizáveis (SOM) a dados censitários portugueses para descobrir tipologias urbanas sem supervisão. Referência de estado da arte em classificação não supervisionada de setores censitários em contexto lusófono.

### Dados e Fontes Institucionais

**[9]** Instituto Brasileiro de Geografia e Estatística  IBGE. (2023). *Censo Demográfico 2022: Cadastro Nacional de Endereços para Fins Estatísticos (CNEFE) Documentação e Metadados*. Rio de Janeiro: IBGE.

> Fonte primária dos dados de 2022. Define o campo `COD_ESPECIE` (8 categorias), `COD_INDICADOR_FINALIDADE_CONST` e a estrutura de `COD_SETOR`. Referência para a interpretação semântica de cada feature do vetor de proporções.

**[10]** Instituto Brasileiro de Geografia e Estatística IBGE. (2011). *Censo Demográfico 2010: Cadastro Nacional de Endereços para Fins Estatísticos (CNEFE)*. Rio de Janeiro: IBGE.

> Fonte primária dos dados de 2010. Define o campo `tipo` de logradouro e `situacao` (1=urbano, 2=rural), utilizado exclusivamente como rótulo de validação *a posteriori*  não como feature de clustering.

**[11]** Instituto Brasileiro de Geografia e Estatística  IBGE. (2022). *Arranjos Populacionais e Concentrações Urbanas do Brasil* (3ª ed.). Rio de Janeiro: IBGE.

> Documenta o crescimento das áreas urbanas e a transição rural-urbana no Brasil no período 2010–2022, corroborando os 3.883 setores Rural→Urbano e os 10.984 setores novos identificados na análise temporal.

### Infraestrutura

**[12]** Raasveldt, M., & Mühleisen, H. (2019). DuckDB: an embeddable analytical database. In *Proceedings of the 2019 International Conference on Management of Data (SIGMOD '19)*, pp. 1981–1984. ACM.

> Artigo original do DuckDB. Descreve a arquitetura de execução SQL colunar e a integração nativa com Parquet  justifica o uso para agregação de 9 milhões de registros sem carregamento completo em RAM.
