import sentence_embedding as st
from typing import List, Optional
import fire
from llama import Dialog, Llama

query = '''You are now given some HLS pragma insertion assistants
assistant 1: int ABij = 0;
product: for(int k = 0; k < M; ++k) {
ABij += A[i][k] * B[k][j];
}
AB[i][j] = ABij;
}
}
}
Figure 7.1: A common three for loop structure for matrix multiplication. The outer for loops,
labeledrowsandcols, iterateacrosstherowsandcolumnsoftheoutputmatrixAB. Theinnermost
loop, labeled product multiplies the appropriate elements of one row of A and one column of B and
accumulates them until it has the result for the element in AB .
Matrix multiplication is a fundamental operation in numerical algorithms. Computing the
product between large matrices can take a significant amount of time. Therefore, it is critically
important part of many of problems in numerical computing. Fundamentally, matrices repre-
sent linear transforms between vector spaces; matrix multiplication provides way to compose the
linear transforms. Applications include linearly changing coordinates (e.g., translation, rotation, assistant 2: [3] Hassan M. Ahmed, Jean-Marc Delosme, and Martin Morf. Highly concurrent computing
structures for matrix arithmetic and signal processing. IEEE Computer, 15(1):65-82, 1982.
[4] Raymond J. Andraka. Building a high performance bit-serial processor in an FPGA. In
Proceedings of Design SuperCon, volume 96, pages 1-5, 1996.
[5] Oriol Arcas-Abella et al. An empirical evaluation of high-level synthesis languages and tools, assistant 3: for (i = N - 1; i > 0; i--) {
shift reg[i] = shift reg[i - 1];
}
shift reg[0] = x;
acc = 0;
MAC:
 Now try your best to Optimize the following code through inserting pragma hls lines

void kernel_bicg(int m,int n,float A[124][116],float s[116],float q[124],float p[116],float r[124])
{
  int i;
  int j;
{
    
    for (i = 0; i < 116; i++) {
      s[i] = ((float )0);
    }
    
    
    
    for (i = 0; i < 124; i++) {
      q[i] = 0.0;
      
      for (j = 0; j < 116; j++) {
        s[j] = s[j] + r[i] * A[i][j];
        q[i] = q[i] + A[i][j] * p[j];
      }
    }
  }
}
'''


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {"role": "system", "content": "Always answer with C code"},
            {"role": "user", "content": query},
        ],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)