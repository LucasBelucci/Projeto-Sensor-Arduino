# Detecção de anomalias pelos dados de um acelerômetro
Capacidade de criar um conjunto caseiro de detecção de vibração de equipamentos, envio de informações via WiFi e análise automática das informações para informar previamente a existência de anomalias.

### Objetivo
Este projeto tem como objetivo a construção de um modelo capaz de captar as frequências de vibração de um equipamento, enviar as informações via wifi para um servidor https, que utilizando uma API em python, é capaz de realizar análises dinâmicas como Distância de Mahalanobis e Análise de Fourier (FFT), fornecendo esses valores para um algoritmo treinado apto a identificar de maneira simplificada se aquelas informações são características de um sistema que está apresentando alguma anomalia ou não.

### Hardware utilizado
- ESP32
- ADXL345

### Esquema de montagem do sistema
![Modelo da conexão dos pinos para o sistema utilizando o microcontrolador ESP32 e o acelerômetro ADXL345](Imagens\Modelo Pinos.png)

### Recursos e Funcionalidades
- Captação da frequência de vibração de funcionamento de equipamentos monitorados
- Envio das informações via WiFi para um servidor HTTPS local
- Extração automática de múltiplas métricas estatísticas e espectrais dos dados do acelerômetro, incluindo:
    - Estatísticas no domínio do tempo (média, desvio padrão, RMS, amplitude, curtose)
    - Correlação entre os eixos (X, Y, Z)
- Análise de frequência via FFT (picos espectrais, energia média, quantidade de harmônicos)
- Realização da Análise de Fourier (FFT) e Distância de Mahalanobis para detecção de anomalias
- Extração de features estatísticas e espectrais para uso com algoritmos de Machine Learning
- Comparação visual de espectros FFT entre estados normais e anômalos

### Tecnologias utilizadas
- Arduino IDE
- Linguagem C++
- Protocolo HTTPS
- Python (Pandas, Numpy, Matplotlib, Seaborn, Scipy, Pandas e Scikit-Learn)
- Jupyter Notebook
- Git/Github
- Pytest (Testes automatizados)

### Tratamento de dados
- OS dados coletados inicialmente foram apenas de vibração, portanto, o único tratamento necessário foi o de separação dos valores de acordo com os eixos captados (X, Y, Z)
- Para a obtenção de algumas métricas, os dados foram selecionados, limpos e randomizados, de modo a garantir o menor overfitting, com a maior precisão.
- Extração de métricas estatísticas e espectrais para alimentar o modelo de ML

### Principais aprendizados
- A utilização de métricas mais eficientes, permitem que o algoritmo utilizado seja otimizado ao ponto de ser possível identificar individualmente cada componente presente no equipamento analisado
- Técnicas de avaliação tais como distância de Mahalonobis são pouco utilizadas na grande maioria de exemplos encontrados pela internet, então ter utilizado, permitiu uma maior compreensão da sua importância e trouxe novos insights com relação a avaliação de algoritmos
- A importância de enriquecer a representação dos dados, combinando estatísticas básicas com análise de frequência (FFT) e correlação entre eixos, o que resultou em um modelo mais inteligente e confiável.
- Criação de uma pipeline de análise de sinais eficientes e extensível


### Testes automatizados
O projeto conta com testes automatizados utilizando pytest, garantindo a confiabilidade das principais funções, são eles:
- Verificação do shape das features extraídas
- Testes de integridade
- Testes com mock de arquivos e funções
- Testes visuais com comparação de espectros

### Exemplo: Comparação de espectros FFT (normal x anomalia)
![Modelo da conexão dos pinos para o sistema utilizando o microcontrolador ESP32 e o acelerômetro ADXL345](Imagens\Modelo Pinos.png)

### Comandos úteis
```
# Instalar dependências
pip install -r requirements.txt

# Executar os testes
pytest -v tests/

```

### Upgrades e melhorias
- A maneira recomendada para aprimoramento desse projeto seria inicialmente de garantir uma conexão segura entre o microcontrolador e o servidor, migração do servidor para a Cloud e cadastro individual dos componentes presentes dentro do equipamento
- Com o descritivo individual de cada componente interno, passa a ser possível a realização de uma previsão de qual anomalia está se desenvolvendo e os componentes candidatos diretamente no espectro apresentado na análise de Fourier, otimizando o tempo e os recursos envolvidos
- Criação de um dashboard que gere em tempo real um acompanhamento dos sensores que estão ativos, mantendo sob supervisão e evitando a necessidade de intervenção fora da programação
- Criação de um sistema de alerta conforme a gravidade das anomalias identificadas, com capacidade de envio das informações para um aplicativo próprio, um grupo de emails e até mesmo números de whatsapp, permitindo uma rápida programação para intervenção ou até mesmo parada imediata, para casos mais críticos, que possam vir a danificar seriamente o equipamento
- Implementada uma nova função de extração de características que amplia significativamente a capacidade do modelo de detectar padrões complexos de anomalias. Essa função combina métricas no domínio do tempo, análise de correlação entre os eixos e espectro de frequência via FFT. Essa abordagem proporciona uma visão mais rica e completa do comportamento do equipamento, aumentando a precisão do modelo de detecção.

### Contribuições
Sinta-se à vontade para fazer adições a esse projeto, enviar sugestões ou relatar bugs

### Agradecimentos
Esse projeto foi derivado inicialmente do Trabalho de Conclusão de Curso desenvolvido no inicio de 2022, porém ao me deparar com um video explicativo no canal do youtube do [Daniel Romero](https://www.youtube.com/watch?v=6MECPST996I&ab_channel=DanielRomero), decidi retomar o projeto e através da explicação apresentada, busquei aprimorar o projeto inicial, já que a primeira ideia era de produção de um conjunto capaz de captar e indicar previamente possíveis anomalias em equipamentos rotativos utilizando um orçamento restrito, contudo, agora com a implementação de um algoritmo inteligente e com novas métricas mais eficientes, conforme foram apresentadas pelo Daniel no seu vídeo, além do aprimoramento de algumas funções e inserções de testes unitários.

### Links Úteis
- [Linkedin](https://www.linkedin.com/in/lucas-belucci/)