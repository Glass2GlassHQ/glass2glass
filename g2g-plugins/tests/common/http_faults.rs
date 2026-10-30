#![allow(dead_code)] // not every test file uses every answer here

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

#[derive(Debug, Clone)]
pub(crate) enum Answer {
    Body(Vec<u8>),
    Status(u16),
    // closes the connection without a response, as an origin that is restarting does
    Drop,
    // sends the headers and the first half of the body, then holds the connection until the client closes it
    Stall(Vec<u8>),
}

// one connection per request, `route` maps a request path to its answer
pub(crate) fn serve_answers(route: impl Fn(&str) -> Answer + Send + 'static) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for mut stream in listener.incoming().flatten() {
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let line = String::from_utf8_lossy(&request);
            let path = line.split_whitespace().nth(1).unwrap_or("");
            match route(path) {
                Answer::Body(body) => {
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&body);
                }
                Answer::Status(status) => {
                    let _ = stream.write_all(
                        format!(
                            "HTTP/1.1 {status} Fault\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )
                        .as_bytes(),
                    );
                }
                Answer::Drop => drop(stream),
                // on its own thread, so the requests after it are answered
                Answer::Stall(body) => {
                    thread::spawn(move || {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(header.as_bytes());
                        let _ = stream.write_all(&body[..body.len() / 2]);
                        let mut byte = [0u8; 1];
                        while stream.read(&mut byte).unwrap_or(0) > 0 {}
                    });
                }
            }
        }
    });
    format!("http://127.0.0.1:{port}")
}
