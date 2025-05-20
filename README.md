# Detecção de anomalias pelos dados de um acelerômetro
Capacidade de criar um conjunto caseiro de detecção de vibração de equipamentos, envio de informações via WiFi e análise automática das informações para informar previamente a existência de anomalias.

### Objetivo
Este projeto tem como objetivo a construção de um modelo capaz de captar as frequências de vibração de um equipamento, enviar as informações via wifi para um servidor https, que utilizando uma API em python, é capaz de realizar análises dinâmicas identificando se as informações possuem indicativos de um sistema com presença de anomalia, para essa análise, será utilizado um algoritmo treinado com o fornecimento de informações de funcionamentos normais e com comportamentos adversos, para que através de cálculos como Distância de Mahalanobis e Análise de Fourier (FFT) seja capaz de realizar a previsão.

### Hardware utilizado
- ESP32
- ADXL345

### Resumo rápido da ligação via I2C:
| Pino do ADXL345 | Pino do ESP32 | Função                              | Descrição técnica                                                                 |
|------------------|----------------|--------------------------------------|-----------------------------------------------------------------------------------|
| VCC              | 3.3V           | Alimentação                         | Fornece tensão de 3.3V para o funcionamento do sensor ADXL345                    |
| GND              | GND            | Terra comum                         | Estabelece referência de zero volts (terra) compartilhada entre os dispositivos  |
| SDA              | GPIO 21        | Dados I2C (Serial Data)             | Linha de dados para comunicação I2C; envia e recebe informações do sensor        |
| SCL              | GPIO 22        | Clock I2C (Serial Clock)            | Linha de clock que sincroniza a comunicação I2C entre ESP32 e sensor             |
| SDO              | GND            | Definição de endereço I2C           | Quando ligado ao GND, define o endereço I2C como 0x53 (alternativa: 0x1D com VCC)|
| CS               | 3.3V           | Seleção de modo de comunicação      | Mantido em nível alto (3.3V) para ativar o modo I2C e desabilitar o modo SPI     |

### Esquema de montagem do sistema
![Modelo da conexão dos pinos para o sistema utilizando o microcontrolador ESP32 e o acelerômetro ADXL345](Imagens/Wiring_scheming.png)

### Sistema montado
![Versão de testes do sistema montado utilizando o microcontrolador ESP32 e o acelerômetro ADXL345](Imagens/Real_circuit.png)

### Recursos e Funcionalidades
- Captação da frequência de vibração de funcionamento de equipamentos monitorados
- Envio das informações via WiFi para um servidor HTTPS local
- Extração de múltiplas métricas estatísticas e espectrais dos dados do acelerômetro, incluindo:
    - Estatísticas no domínio do tempo (média, desvio padrão, RMS, amplitude, curtose)
    - Correlação entre os eixos (X, Y, Z)
- Análise de frequência via FFT (picos espectrais, energia média, quantidade de harmônicos)
- Realização da Análise de Fourier (FFT) e Distância de Mahalanobis para detecção de anomalias
- Utilização das features estatísticas e espectrais para treinamento algoritmos de Machine Learning
- Comparação visual de espectros FFT entre estados normais e anômalos
- Determinação da distância de Mahalanobis como critério para indicativo de anomalia
- Criação de testes automatizados para testar todas as funções utilizadas durante o treinamento, tanto para obtenção das estátisticas, quanto a plotagem dos gráficos

### Tecnologias utilizadas
- Arduino IDE
- Linguagem C++
- Protocolo HTTPS
- Python (Pandas, Numpy, Matplotlib, Seaborn, Scipy, Pandas e Scikit-Learn)
- Jupyter Notebook
- Git/Github
- Pytest (Testes automatizados)

### Coleta dos dados
## ESP32
Lógica criada através da interface do ArduinoIDE, faz uma checagem inicial dos componentes, posteriormente uma verificação de conexão com o servidor HTTPS e finalmente habilita um timer para a realização de maneira periódica a coleta dos dados fornecidos pelo sensor ADXL345 e envio para um servidor HTTPS.

## Servidor
Servidor simples feito em python com apenas o método POST para receber e posteriormente agrupar as primeiras 200 coletas realizadas pelo sensor ADXL345.

## Criação dos dados
Os arquivos recebidos oriundos do ESP32 são agrupados e salvos em um arquivos CSV no diretório característico.

### Tratamento de dados
- OS dados coletados inicialmente foram apenas de vibração, portanto, o único tratamento necessário foi o de separação dos valores de acordo com os eixos captados (X, Y, Z)
- Para a obtenção de algumas métricas, os dados foram selecionados, limpos e randomizados, de modo a garantir o menor overfitting, com a maior precisão.
- Extração de métricas estatísticas e espectrais para alimentar o modelo de ML

### Confiança na detecção de anomalias e suavização
O sistema calcula uma métrica de confiança associada à detecção de anomalias, baseada principalmente na Distância de Mahalanobis em relação ao limiar definido.

#### Como a confiança é calculada?
Quando a distância é muito superior ao limiar, a confiança tende a 100%, indicando alta certeza da anomalia, quando a distância é próxima ao limiar, a confiança diminui, indicando incerteza.

#### Desafios observados
Durante testes, foi identificado que a confiança apresentava oscilações abruptas entre amostras, como resultado de uma inserção não continua dos dados, uma vez que, caso os dados estivessem sendo observados em intervalos constantes a falha iria progredir de modo gradual e não sofrer alterações abruptas, pensando em solucionar essas flutuações, que em cenários de falhas podem causar falsas impressões de recuperação ou de agravamento repentino, foi implementada uma suavização exponencial.

#### Solução aplicada: suavização exponencial
Foi implementada uma técnica de suavização exponencial da confiança:

```
confidence_smooth = alpha * confidence_atual + (1 - alpha) * confidence_smooth_anterior
```

Onde alpha controla o grau de suavização.

#### Benefícios práticos observados
- Tornou a confiança uma métrica mais confiável de tendência.
- Reduziu alarmes falsos em situações de oscilações naturais do sistema.
- Permitiu observar claramente a evolução progressiva de degradação quando presente.

### Arquivos e módulos principais

streamlit_app.py: 
- Responsável por fornecer uma interface visual interativa e em tempo real, onde:
- Os dados coletados do acelerômetro são processados ao vivo.
- O modelo é aplicado em tempo real.
- Exibe as principais métricas (distância, limiar, confiança).
- Mostra visualizações interativas como espectros FFT.
- Permite acompanhar ao vivo o comportamento dos sensores e o status de anomalia.

utils.py:
- Módulo central que organiza todas as funções reutilizáveis do projeto, incluindo:
- Extração de features estatísticas e espectrais dos dados do acelerômetro.
- Cálculo da distância de Mahalanobis.
- Função de detecção de anomalia com cálculo de confiança e suavização integrada.
- Funções de visualização gráfica (espectros FFT, distribuição das distâncias, etc).
- Carregamento do modelo e do scaler.
- Atualização dinâmica de confiança suavizada ao longo do tempo.
- Este módulo foi estruturado para permitir máxima reutilização, clareza e expansão futura.

training.py:
- Responsável pela realização do treinamento e criação do modelo utilizado para análise posterior, além da determinação de critérios essenciais como o limiar da distância de Mahalonobis, fundamental para classificação na presença de anomalia.

analysis.py:
- Módulo principal para organização das funções utilizadas para obtenção das métricas estatísticas, gráficos gerados, comparações de resultados, amplamente aplicado durante o training.

test_features.py:
- Algoritmo responsável pela realização dos testes automatizados das funções presentes majoritariamente em analysis.

server.py
- Dois arquivos diferentes, porém com o mesmo nome, o primeiro é utilizado para criação do server HTTPS que irá receber e agrupar as informações durante a coleta dos dados realizadas pelo sensor ADXL345 e passadas ao ESP32.
- O segundo é responsável pela criação do servidor que irá hospedar o streamlit, e portanto, possuir o método POST que recebe o arquivo CSV com os dados coletados, aplica o algoritmo treinado e retorna o resultado da distância de Mahalanobis.

### Principais aprendizados
- A utilização de métricas mais eficientes, permitem que o algoritmo utilizado seja otimizado ao ponto de ser possível identificar individualmente cada componente presente no equipamento analisado.
- Técnicas de avaliação tais como distância de Mahalonobis permitiu uma maior compreensão da sua importância e trouxe novos insights com relação a avaliação de algoritmos.
- A importância de enriquecer a representação dos dados, combinando estatísticas básicas com análise de frequência (FFT) e correlação entre eixos, o que resultou em um modelo mais inteligente e confiável.
- Aprimoramento do algoritmo de aprendizado através da utilização de métricas estatísticas.
- Criação de uma pipeline de análise de sinais eficientes e extensível.
- Criação de um algoritmo de testes automatizados.
- Criação de um dashboard em streamlit.


### Testes automatizados
O projeto conta com testes automatizados utilizando pytest, garantindo a confiabilidade das principais funções, são eles:
- Verificação do shape das features extraídas
- Testes de integridade
- Testes com mock de arquivos e funções
- Testes visuais com comparação de espectros

### Exemplo: Comparação de espectros FFT (normal x anomalia)
![Comparação dos espectros FFT entre as amostrais normais e anormais](Imagens/fft_comparison_20250418_185547.png)

### Comandos úteis
```
# Instalar dependências
pip install -r requirements.txt

# Executar os testes
python -m pytest -v ProjetoSensor/tests

```

### Upgrades e melhorias
- A maneira recomendada para aprimoramento desse projeto seria inicialmente de garantir uma conexão segura entre o microcontrolador e o servidor, migração do servidor para a Cloud e cadastro individual dos componentes presentes dentro do equipamento.
- Com o descritivo individual de cada componente interno, passa a ser possível a realização de uma previsão de qual anomalia está se desenvolvendo e os componentes candidatos diretamente no espectro apresentado na análise de Fourier, otimizando o tempo e os recursos envolvidos.
- Criação de um dashboard que gere em tempo real um acompanhamento dos sensores que estão ativos, mantendo sob supervisão e evitando a necessidade de intervenção fora da programação.
- Substituição do envio de informações de pacote de arquivos para acompanhamento em tempo real, permitindo inicio das rotinas de coleta, acompanhamento da progressão da anomalia.
- Criação de um sistema de alerta conforme a gravidade das anomalias identificadas, com capacidade de envio das informações para um aplicativo próprio, um grupo de emails e até mesmo números de whatsapp, permitindo uma rápida programação para intervenção ou até mesmo parada imediata, para casos mais críticos, que possam vir a danificar seriamente o equipamento.
- Implementada uma nova função de extração de características que amplia significativamente a capacidade do modelo de detectar padrões complexos de anomalias. Essa função combina métricas no domínio do tempo, análise de correlação entre os eixos e espectro de frequência via FFT. Essa abordagem proporciona uma visão mais rica e completa do comportamento do equipamento, aumentando a precisão do modelo de detecção.

### Contribuições
Sinta-se à vontade para fazer adições a esse projeto, enviar sugestões ou relatar bugs.

### Agradecimentos
Esse projeto foi derivado inicialmente do Trabalho de Conclusão de Curso desenvolvido no inicio de 2022, porém ao me deparar com um video explicativo no canal do youtube do [Daniel Romero](https://www.youtube.com/watch?v=6MECPST996I&ab_channel=DanielRomero), decidi retomar o projeto e através da explicação apresentada, busquei aprimorar o projeto inicial, já que a primeira ideia era de produção de um conjunto capaz de captar e indicar previamente possíveis anomalias em equipamentos rotativos utilizando um orçamento restrito, contudo, agora com a implementação de um algoritmo inteligente e com novas métricas mais eficientes, conforme foram apresentadas pelo Daniel no seu vídeo, além do aprimoramento de algumas funções e inserções de testes unitários.
Para fins educacionais, apesar de ter realizado a coleta de dados, optou-se por utilizar os mesmos dados apresentados pelo Daniel, já que permitiria uma comparação, e portanto, ajudaria no momento em que deseja-se realizar um aprimoramento em alguns aspectos do sistema.

### Links Úteis
- [Linkedin](https://www.linkedin.com/in/lucas-belucci/)
