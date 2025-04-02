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
- Geração de diferentes tipos de estatísticas (média, variância, curtose)
- Realização da Análise de Fourier (FFT) e Distância de Mahalanobis para detecção de anomalias 

### Tecnologias utilizadas
- Arduino IDE
- Linguagem C++
- Protocolo HTTPS
- Python (Pandas, Numpy, Matplotlib, Seaborn, Scipy, Pandas e Scikit-Learn)
- Jupyter Notebook
- Git/Github

### Principais aprendizados
- A utilização de métricas mais eficientes, permitem que o algoritmo utilizado seja otimizado ao ponto de ser possível identificar individualmente cada componente presente no equipamento analisado
- Técnicas de avaliação tais como distância de Mahalonobis são pouco utilizadas na grande maioria de exemplos encontrados pela internet, então ter utilizado, permitiu uma maior compreensão da sua importância e trouxe novos insights com relação a avaliação de algoritmos
- Apesar da complexidade de métricas apresentadas durante o projeto, as fundamentais foram as estatísticas básicas em conjunto com a análise de Fourier (FFT), permitindo a identificação com clareza das anomalias

### Tratamento de dados
- OS dados coletados inicialmente foram apenas de vibração, portanto, o único tratamento necessário foi o de separação dos valores de acordo com os eixos captados (X, Y, Z)
- Para a obtenção de algumas métricas, os dados foram selecionados, limpos e randomizados, de modo a garantir o menor overfitting, com a maior precisão 

### Upgrades e melhorias
- A maneira recomendada para aprimoramento desse projeto seria inicialmente de garantir uma conexão segura entre o microcontrolador e o servidor, migração do servidor para a Cloud e cadastro individual dos componentes presentes dentro do equipamento
- Com o descritivo individual de cada componente interno, passa a ser possível a realização de uma previsão de qual anomalia está se desenvolvendo e os componentes candidatos diretamente no espectro apresentado na análise de Fourier, otimizando o tempo e os recursos envolvidos
- Criação de um dashboard que gere em tempo real um acompanhamento dos sensores que estão ativos, mantendo sob supervisão e evitando a necessidade de intervenção fora da programação
- Criação de um sistema de alerta conforme a gravidade das anomalias identificadas, com capacidade de envio das informações para um aplicativo próprio, um grupo de emails e até mesmo números de whatsapp, permitindo uma rápida programação para intervenção ou até mesmo parada imediata, para casos mais críticos, que possam vir a danificar seriamente o equipamento

### Contribuições
Sinta-se à vontade para fazer adições a esse projeto, enviar sugestões ou relatar bugs

### Agradecimentos
Esse projeto foi derivado inicialmente do Trabalho de Conclusão de Curso desenvolvido no inicio de 2022, porém ao me deparar com um video explicativo no canal do youtube do [Daniel Romero](https://www.youtube.com/watch?v=6MECPST996I&ab_channel=DanielRomero), decidi retomar o projeto e através da explicação apresentada, busquei aprimorar o projeto inicial, já que a primeira ideia era de produção de um conjunto capaz de captar e indicar previamente possíveis anomalias em equipamentos rotativos utilizando um orçamento restrito, contudo, agora com a implementação de um algoritmo inteligente e com novas métricas mais eficientes, conforme foram apresentadas pelo Daniel no seu vídeo

### Links Úteis
- [Linkedin](https://www.linkedin.com/in/lucas-belucci/)