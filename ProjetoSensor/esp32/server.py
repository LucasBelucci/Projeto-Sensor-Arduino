import json
import argparse
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class WebRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, output_dir, *args, **kwargs):
        self.output_dir = output_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"1") # Retorna 1 para indicar que está pronto

    def do_POST(self):
        try:
            # Leitura e separação dos dados
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            sensor_data = json.loads(post_data.decode('utf-8'))

            # Definição do diretório e do nome do arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(self.output_dir) / f'sensor_data_{timestamp}.csv'

            # Salvando as informações no CSV
            self._save_data_to_csv(sensor_data, filepath)

            self.send_response(204) # Retornando 204 para indicar que foi feito o salvamento

        except Exception as e:
            print(f"Error processing data: {str(e)}")
            self.send_response(500)
        self.end_headers()

    def _save_data_to_csv(self, data, filepath):
        with open(filepath, "w") as f:
            num_samples = len(data["x"])
            for i in range(num_samples):
                f.write(f"{data['x'][i]}, {data['y'][i]}, {data['z'][i]}\n")
    
def create_server(output_dir, port):

    # Verifica se o diretório existe, caso não, ele é criado
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Cria um handler e passa as suas configurações
    def handler(*args, **kwargs):
        return WebRequestHandler(output_dir, *args, **kwargs)
    
    # Cria e retorna um servidor
    return HTTPServer(("", port), handler)

def main():
    # Define alguns padrões para as linhas de comando
    parser = argparse.ArgumentParser(description="Sensor Data Collection Server")
    
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="sensor_data"
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=4242
    )

    args = parser.parse_args()

    # Cria e inicializa u mservidor
    server = create_server(args.dir, args.port)

    # Mensagem de inicialização do servidor
    print("\nSensor Data Collection Server")
    print(f"Saving data to: {args.dir}")
    print(f"Server running on port: {args.port}")
    print("Press CTRL+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()