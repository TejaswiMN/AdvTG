"""Build train_data2.json (raw HTTP requests + labels) from CIC-IDS2017.

The Kaggle set `chethuhn/network-intrusion-dataset` ships two things:
  * MachineLearningCVE/*.csv -- per-flow numeric features + a Label column
  * PCAPs/*.pcap             -- the raw capture the CSVs were derived from

DL/main.py needs the *text* of HTTP requests, which exists only in the pcaps,
so we extract requests from the pcaps and join the CSV Label onto them via the
flow 5-tuple.

Usage:
    python scripts/build_train_data.py \
        --pcap-dir  /path/to/PCAPs \
        --csv-dir   /path/to/TrafficLabelling \
        --out       dataset/train_data2.json

Backends: tshark (preferred, handles TCP reassembly) or scapy (--backend scapy).
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

METHODS = (b"GET", b"POST", b"HEAD", b"PUT", b"DELETE", b"OPTIONS",
           b"PATCH", b"TRACE", b"CONNECT")


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def load_label_index(csv_dir):
    """(src, sport, dst, dport) -> 'Malicious' | 'Benign', from the flow CSVs."""
    import pandas as pd

    index = {}
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        raise SystemExit("no CSVs found in %s" % csv_dir)
    for path in files:
        # CICIDS2017 headers carry stray spaces and a few non-utf8 bytes.
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        need = ["Source IP", "Source Port", "Destination IP",
                "Destination Port", "Label"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            # MachineLearningCVE csvs drop the IP columns; only TrafficLabelling
            # keeps them. Such a file cannot be joined onto packets.
            print("  ! %s: missing %s, skipped" % (os.path.basename(path), missing),
                  file=sys.stderr)
            continue
        for src, sport, dst, dport, label in zip(
                df["Source IP"], df["Source Port"], df["Destination IP"],
                df["Destination Port"], df["Label"]):
            try:
                key = (str(src), int(sport), str(dst), int(dport))
            except (ValueError, TypeError):
                continue
            label = str(label).strip()
            value = "Benign" if label.upper() == "BENIGN" else "Malicious"
            # A 5-tuple recurs across the day; malicious wins so attacks that
            # reuse a port are not diluted into the benign class.
            if index.get(key) != "Malicious":
                index[key] = value
        print("  loaded %s (%d flows)" % (os.path.basename(path), len(df)))
    return index


def lookup_label(index, src, sport, dst, dport):
    return (index.get((src, sport, dst, dport))
            or index.get((dst, dport, src, sport)))


# Hosts CIC used to launch the 2017 attacks: the Kali box on the outside
# (205.174.165.73) reached through the firewall's NAT addresses, plus the
# internal attacker used on the infiltration day.
DEFAULT_ATTACK_IPS = ["205.174.165.73", "205.174.165.80", "205.174.165.69",
                      "205.174.165.70", "205.174.165.71", "172.16.0.1",
                      "192.168.10.8"]


# --------------------------------------------------------------------------
# request parsing
# --------------------------------------------------------------------------
def parse_request(raw):
    """Split one raw HTTP request into (request line, headers dict, body)."""
    head, sep, body = raw.partition("\r\n\r\n")
    if not sep:
        head, sep, body = raw.partition("\n\n")
    lines = head.replace("\r\n", "\n").split("\n")
    if not lines or not lines[0].strip():
        return None
    headers = {}
    for line in lines[1:]:
        name, colon, value = line.partition(":")
        if colon:
            headers[name.strip()] = value.strip()
    return lines[0].strip(), headers, body


# --------------------------------------------------------------------------
# tshark backend
# --------------------------------------------------------------------------
def iter_requests_tshark(pcap, tshark="tshark"):
    """Yield (src, sport, dst, dport, raw_request) using tshark reassembly."""
    fields = ["ip.src", "ip.dst", "tcp.srcport", "tcp.dstport",
              "http.request.line", "http.request.method", "http.request.uri",
              "http.request.version", "http.file_data"]
    cmd = [tshark, "-r", pcap, "-Y", "http.request", "-T", "ek"]
    for f in fields:
        cmd += ["-e", f]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=1 << 20)
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line or b'"layers"' not in line:
                continue
            try:
                layers = json.loads(line)["layers"]
            except (ValueError, KeyError):
                continue

            def one(name, default=""):
                v = layers.get(name)
                if isinstance(v, list):
                    return v[0] if v else default
                return default if v is None else v

            method = one("http_http_request_method")
            uri = one("http_http_request_uri")
            version = one("http_http_request_version") or "HTTP/1.1"
            if not method:
                continue
            header_lines = layers.get("http_http_request_line") or []
            if isinstance(header_lines, str):
                header_lines = [header_lines]
            body = one("http_http_file_data")
            if body and ":" in body and " " not in body:
                # tshark hands file_data back as colon-separated hex bytes.
                try:
                    body = bytes.fromhex(body.replace(":", "")).decode(
                        "utf-8", "replace")
                except ValueError:
                    pass
            raw = ("%s %s %s\r\n" % (method, uri, version)
                   + "".join(header_lines) + "\r\n" + (body or ""))
            yield (one("ip_src"), int(one("tcp_srcport", 0) or 0),
                   one("ip_dst"), int(one("tcp_dstport", 0) or 0), raw)
    finally:
        proc.stdout.close()
        proc.wait()


# --------------------------------------------------------------------------
# scapy backend
# --------------------------------------------------------------------------
def iter_requests_scapy(pcap):
    """Yield requests by reassembling client->server TCP payloads with scapy."""
    from scapy.all import PcapReader, IP, TCP

    streams = defaultdict(dict)   # flow key -> {seq: payload}
    order = []
    with PcapReader(pcap) as reader:
        for pkt in reader:
            if IP not in pkt or TCP not in pkt:
                continue
            payload = bytes(pkt[TCP].payload)
            if not payload:
                continue
            key = (pkt[IP].src, int(pkt[TCP].sport),
                   pkt[IP].dst, int(pkt[TCP].dport))
            if key not in streams:
                order.append(key)
            streams[key].setdefault(pkt[TCP].seq, payload)

    for key in order:
        chunks = streams[key]
        buf = b"".join(chunks[s] for s in sorted(chunks))
        if not buf.startswith(METHODS):
            continue
        for raw in split_pipelined(buf):
            yield key[0], key[1], key[2], key[3], raw


def split_pipelined(buf):
    """Split a client-side stream buffer into individual requests."""
    out = []
    while buf.startswith(METHODS):
        end = buf.find(b"\r\n\r\n")
        sep = 4
        if end == -1:
            end, sep = buf.find(b"\n\n"), 2
        if end == -1:
            out.append(buf.decode("utf-8", "replace"))
            break
        head = buf[:end]
        length = 0
        for line in head.split(b"\n"):
            if line.lower().startswith(b"content-length:"):
                try:
                    length = int(line.split(b":", 1)[1].strip())
                except ValueError:
                    length = 0
        stop = end + sep + length
        out.append(buf[:stop].decode("utf-8", "replace"))
        buf = buf[stop:]
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pcap-dir", required=True, help="directory of .pcap files")
    ap.add_argument("--csv-dir", help="directory of labelled flow CSVs "
                                      "(TrafficLabelling, which keeps the IP columns)")
    ap.add_argument("--out", default="dataset/train_data2.json")
    ap.add_argument("--backend", choices=["auto", "tshark", "scapy"], default="auto")
    ap.add_argument("--tshark", default="tshark", help="path to the tshark binary")
    ap.add_argument("--unmatched", choices=["skip", "benign"], default="skip",
                    help="what to do with requests no CSV flow labels")
    ap.add_argument("--attack-ips", nargs="*", default=None,
                    metavar="IP",
                    help="label by attacker IP instead of / in addition to the "
                         "CSV join; pass with no values to use the known "
                         "CIC-IDS2017 attacker hosts. Needed when your CSVs are "
                         "the MachineLearningCVE ones, which have no IP columns.")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="cap records per label (0 = no cap)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "tshark" if shutil.which(args.tshark) else "scapy"
    print("backend: %s" % backend)

    attack_ips = None
    if args.attack_ips is not None:
        attack_ips = set(args.attack_ips or DEFAULT_ATTACK_IPS)
        print("attacker IPs: %s" % ", ".join(sorted(attack_ips)))

    index = {}
    if args.csv_dir:
        print("loading flow labels...")
        index = load_label_index(args.csv_dir)
        print("  %d labelled flows" % len(index))
        if not index and attack_ips is None:
            raise SystemExit(
                "no CSV yielded a usable flow index (the MachineLearningCVE "
                "files have no IP columns). Either point --csv-dir at the "
                "TrafficLabelling / GeneratedLabelledFlows CSVs, or rerun with "
                "--attack-ips to label by attacker host.")
    elif attack_ips is None and args.unmatched == "skip":
        raise SystemExit("pass --csv-dir, --attack-ips, or --unmatched benign")

    pcaps = sorted(glob.glob(os.path.join(args.pcap_dir, "*.pcap"))
                   + glob.glob(os.path.join(args.pcap_dir, "*.pcapng")))
    if not pcaps:
        raise SystemExit("no pcaps found in %s" % args.pcap_dir)

    records, counts = [], Counter()
    for pcap in pcaps:
        source = os.path.splitext(os.path.basename(pcap))[0]
        print("parsing %s ..." % source)
        stream = (iter_requests_tshark(pcap, args.tshark) if backend == "tshark"
                  else iter_requests_scapy(pcap))
        kept = 0
        for src, sport, dst, dport, raw in stream:
            label = lookup_label(index, src, sport, dst, dport)
            if label is None and attack_ips is not None:
                label = ("Malicious" if src in attack_ips or dst in attack_ips
                         else "Benign")
            if label is None:
                if args.unmatched == "skip":
                    counts["unmatched"] += 1
                    continue
                label = "Benign"
            parsed = parse_request(raw)
            if parsed is None:
                continue
            line, headers, body = parsed
            records.append({
                "Request Line": line,
                "Request Headers": headers,
                "Request Body": body,
                "Label": label,
                "Source": source,
            })
            counts[label] += 1
            kept += 1
        print("  %d requests" % kept)

    if args.max_per_class:
        random.seed(args.seed)
        by_label = defaultdict(list)
        for rec in records:
            by_label[rec["Label"]].append(rec)
        records = []
        for label, group in by_label.items():
            random.shuffle(group)
            records.extend(group[:args.max_per_class])

    random.seed(args.seed)
    random.shuffle(records)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print("\nwrote %d records to %s" % (len(records), args.out))
    print("  %s" % dict(Counter(r["Label"] for r in records)))
    if counts["unmatched"]:
        print("  %d requests dropped (no CSV flow match)" % counts["unmatched"])


if __name__ == "__main__":
    main()
