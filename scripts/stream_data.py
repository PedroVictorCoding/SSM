#!/usr/bin/env python3
"""
Progressive Dataset Streamer for SAIL.

Slowly and continuously retrieves or generates text chunks and drops them
into `data/incoming/` so the SAIL continual trainer progressively ingests them
and updates the living model.

Usage:
  python3 scripts/stream_data.py --incoming data/incoming --interval 10 --max-chunks 20
"""

import argparse
import os
import time
import uuid
from datetime import datetime

CURATED_KNOWLEDGE_STREAM = [
    # Science & Physics
    """The theory of general relativity, formulated by Albert Einstein in 1915, explains gravity 
not as an invisible force between masses, but as a consequence of the curvature of spacetime caused 
by the uneven distribution of mass and energy. The most extreme spacetime curvature occurs in black 
holes, regions from which nothing, not even light, can escape.""",

    # Computer Science & Neural Networks
    """State Space Models (SSMs) and Mamba represent a paradigm shift in sequence modeling. Unlike 
traditional Transformer architectures that suffer from quadratic computational complexity O(L^2) with 
respect to context length, selective SSMs achieve linear-time training O(L) and constant O(1) memory 
inference per token. This enables long-sequence modeling and fast real-time decoding.""",

    # History & Philosophy
    """During the Renaissance in 14th to 17th century Europe, a cultural rebirth bridged the Middle 
Ages to modernity. Humanism emerged, emphasizing individual potential, critical inquiry, empirical 
observation, and the recovery of classical philosophy and literature from ancient Greece and Rome.""",

    # Astronomy & Planetary Science
    """The James Webb Space Telescope (JWST) operates primarily in the infrared spectrum, allowing it 
to observe the earliest galaxies formed shortly after the Big Bang, peer through dense interstellar dust 
clouds where planetary systems are born, and analyze the atmospheric composition of distant exoplanets.""",

    # Biology & Evolution
    """Deoxyribonucleic acid (DNA) carries the genetic instructions for the development, functioning, 
growth, and reproduction of all known organisms. The double-helix structure, deciphered by Watson, Crick, 
Franklin, and Wilkins, consists of two complementary polynucleotide chains formed of nucleotide bases: 
adenine, thymine, guanine, and cytosine.""",

    # Economics & Modern Systems
    """Modern macroeconomic theory explores the behavior of aggregate economies, analyzing inflation, 
unemployment, monetary policy, and fiscal dynamics. Central banks regulate the money supply and interest 
rates to promote price stability and sustainable economic expansion.""",

    # Technology & Rust Systems
    """The Rust programming language achieves memory safety without a garbage collector through its 
ownership, borrowing, and lifetime system. By enforcing aliasing XOR mutability at compile time, Rust 
eliminates data races, null pointer dereferences, and buffer overflows before code runs in production.""",

    # Art & Literature
    """William Shakespeare's dramatic works encompass tragedies, comedies, and histories. In 'Hamlet', 
themes of mortality, betrayal, madness, and revenge are explored through philosophical soliloquies 
that examine the human consciousness and the burden of moral choices.""",
]

def main():
    parser = argparse.ArgumentParser(description="Progressive Dataset Streamer for SAIL")
    parser.add_argument("--incoming", default="data/incoming", help="Incoming directory to drop text files")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between data drops")
    parser.add_argument("--max-chunks", type=int, default=0, help="Maximum chunks to drop (0 = infinite)")
    args = parser.parse_args()

    os.makedirs(args.incoming, exist_ok=True)
    print(f"[Streamer] Active. Dropping data into {args.incoming} every {args.interval}s...")

    chunk_idx = 0
    while True:
        chunk_idx += 1
        text = CURATED_KNOWLEDGE_STREAM[(chunk_idx - 1) % len(CURATED_KNOWLEDGE_STREAM)]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chunk_{timestamp}_{uuid.uuid4().hex[:6]}.txt"
        filepath = os.path.join(args.incoming, filename)
        
        temporary = filepath + ".tmp"
        with open(temporary, "x", encoding="utf-8") as f:
            f.write(text.strip())
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, filepath)
            
        print(f"[Streamer] [{chunk_idx}] Dropped {len(text)} chars to {filepath}")
        
        if args.max_chunks > 0 and chunk_idx >= args.max_chunks:
            print(f"[Streamer] Reached maximum chunk limit ({args.max_chunks}). Exiting.")
            break
            
        time.sleep(args.interval)

if __name__ == "__main__":
    main()

