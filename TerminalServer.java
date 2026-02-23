import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.Executors;

/**
 * Tiny embedded server to preview the faux-terminal WebUI.
 * Run from repo root: `javac TerminalServer.java && java TerminalServer`
 * Then open http://localhost:8080
 */
public class TerminalServer {

    private static final int PORT = 8080;
    private static final Path INDEX = Path.of("web/index.html");

    public static void main(String[] args) throws IOException {
        if (!Files.exists(INDEX)) {
            System.err.println("No se encuentra " + INDEX.toAbsolutePath());
            System.exit(1);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/", TerminalServer::serveIndex);
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();

        System.out.println("Terminal UI lista en http://localhost:" + PORT);
    }

    private static void serveIndex(HttpExchange exchange) throws IOException {
        byte[] body = Files.readAllBytes(INDEX);
        exchange.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
        exchange.sendResponseHeaders(200, body.length);
        exchange.getResponseBody().write(body);
        exchange.close();
    }
}
