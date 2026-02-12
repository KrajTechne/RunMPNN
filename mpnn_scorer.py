import os
import subprocess
import torch
import numpy as np

import os
import subprocess
import torch

class MPNNScorer:
    """
    A specific wrapper for the MPNN scoring protocol
    """
    def __init__(self, mpnn_weights_path: str,  score_py_path: str ="RunMPNN/score.py", python_path:str ="python"):
        self.score_py = score_py_path
        self.python = python_path
        self.mpnn_weights_path = mpnn_weights_path
 
    def score(
        self,
        pdb_path,
        out_folder,
        model_type="soluble_mpnn",
        chains_to_score=None,
        parse_these_chains_only=None,
        fixed_residues=None,
        redesigned_residues=None,
        use_atom_context=True,
        use_side_chain_context=False,
        single_aa_score = True,
        use_sequence = True,
        batch_size= 1,
        number_of_batches=10,
        seed=111,
        verbose=False
    ):
        """
        Runs the scoring protocol to evaluate sequence probability/solubility.
        
        Args:
            pdb_path (str): Path to the input PDB file.
            out_folder (str): Directory to save results.
            model_type (str): 'soluble_mpnn', 'ligand_mpnn', etc.
            
            chains_to_score (list/str): Chains to score (e.g. ['A','B']). Other chains are fixed context.
            parse_these_chains_only (list/str): Only parse these chains from the PDB (e.g. ['A']).
            
            fixed_residues (list/str): Residues to freeze (e.g. ['A1', 'A2']).
            redesigned_residues (list/str): Residues to explicitly score (e.g. ['A10']).
            
            use_atom_context (bool): If True, uses non-protein atoms (ligands/DNA) as context.
            use_side_chain_context (bool): If True, uses side-chains of fixed residues as context.

            single_aa_score (bool): If True, computes single aa probabilities: p(AA_1|backbone, AA_{all except AA_1}), p(AA_2|backbone, AA_{all except AA_2})
            use_sequence (bool): If True, uses sequence as context. Otherwise, uses only the protein backbone as context
            
            batch_size (int): Sequences processed per pass (usually 1 for scoring).
            number_of_batches (int): Number of random decoding paths to average (default 10).
            seed (int): Random seed.
            verbose (bool): Print subprocess output.
            
        Returns:
            dict: The loaded results from the .pt file (probs, logits, etc.).
        """
        # Ensure absolute paths
        pdb_path = os.path.abspath(pdb_path)
        out_folder = os.path.abspath(out_folder)
        os.makedirs(out_folder, exist_ok=True)

        # Build Base Command
        cmd = [
            self.python, self.score_py,
            "--pdb_path", pdb_path,
            "--out_folder", out_folder,
            "--model_type", model_type,
            "--seed", str(seed), 
            "--use_sequence", "1" if use_sequence else "0",            
            "--batch_size", str(batch_size),
            "--number_of_batches", str(number_of_batches),
            "--verbose", "1" if verbose else "0"
        ]
       

        # If want to do single_aa_score, so essentially MLM. Condition on backbone and all surrounding amino acids except one being predicted
        customization_args = {}
        if single_aa_score:
            cmd += ["--single_aa_score", "1"]
            customization_args["single_aa_score"] = True
            customization_args["autoregressive_score"] = False
        # Otherwise, MPNN will utilize autoregressive scoring. Condiction on backbone and previous amino acid that was predicted
        else:
            cmd += ["autoregressive_score", "1"]
            customization_args["single_aa_score"] = False
            customization_args["autoregressive_score"] = True
        
        # --- Model checkpoint handling ---
        # Ensure user can provide appropriate path to MPNN model weights
        # Situation #1: Assumption MPNN weights are stored in model_params which is a subfolder of current working directory where score.py is located
        if self.mpnn_weights_path == "":
            score_py_path = os.path.abspath(self.score_py)
            BASE_DIR = os.path.dirname(score_py_path)
            MODEL_DIR = os.path.join(BASE_DIR, "model_params")
        
        # Situation #2: User provides absolute path to model weights folder
        else:
            MODEL_DIR = os.path.abspath(self.mpnn_weights_path)

        if model_type == "protein_mpnn":
            cmd += ["--checkpoint_protein_mpnn", os.path.join(MODEL_DIR, "proteinmpnn_v_48_020.pt")]
        elif model_type == "ligand_mpnn":
            cmd += ["--checkpoint_ligand_mpnn", os.path.join(MODEL_DIR, "ligandmpnn_v_32_010_25.pt")]
        elif model_type == "soluble_mpnn":
            cmd += ["--checkpoint_soluble_mpnn", os.path.join(MODEL_DIR, "solublempnn_v_48_020.pt")]

        # --- Handle Optional Arguments ---

        # 1. Chains to Score (Comma-separated)
        # Maps to --chains_to_design in score.py
        if chains_to_score:
            if isinstance(chains_to_score, (list, tuple)):
                chains_to_score = ",".join(chains_to_score)
            cmd += ["--chains_to_design", chains_to_score]

        # 2. Parse Specific Chains (Comma-separated)
        if parse_these_chains_only:
            if isinstance(parse_these_chains_only, (list, tuple)):
                parse_these_chains_only = ",".join(parse_these_chains_only)
            cmd += ["--parse_these_chains_only", parse_these_chains_only]

        # 3. Fixed Residues (Space-separated)
        if fixed_residues:
            if isinstance(fixed_residues, (list, tuple)):
                fixed_residues = " ".join(fixed_residues)
            cmd += ["--fixed_residues", fixed_residues]

        # 4. Redesigned Residues (Space-separated)
        if redesigned_residues:
            if isinstance(redesigned_residues, (list, tuple)):
                redesigned_residues = " ".join(redesigned_residues)
            cmd += ["--redesigned_residues", redesigned_residues]

        # 5. Context Flags
        cmd += ["--ligand_mpnn_use_atom_context", "1" if use_atom_context else "0"]
        cmd += ["--ligand_mpnn_use_side_chain_context", "1" if use_side_chain_context else "0"]

        # --- Execution ---
        
        if verbose:
             # Create a dictionary of the settings you want to inspect
            config = {
                "pdb_path": pdb_path,
                "out_folder": out_folder,
                "model_type": model_type,
                "parse_these_chains_only": parse_these_chains_only,
                "chains_to_score": chains_to_score,
                "fixed_residues": fixed_residues,
                "redesigned_residues": redesigned_residues,
                "batch_size": batch_size,
                "number_of_batches": number_of_batches,
                "seed": seed,
                "use_atom_context": use_atom_context,
                "use_side_chain_context": use_side_chain_context
            }
            config.update(customization_args)
            print("="*40)
            print("MPNN Scoring Configuration:")
            for k, v in config.items():
                # Handle None values gracefully for printing
                val_str = str(v) if v is not None else "None"
                print(f"{k:30}: {val_str}")
            print("="*40)

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Scoring failed with error:\n{result.stderr}")

        # --- Retrieve Data ---
        
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
        print(f"Output saved to: {out_folder}/{pdb_name}.pt")
        # score.py saves as: {out_folder}/{pdb_name}.pt
        output_pt_path = os.path.join(out_folder, f"{pdb_name}.pt")

        if not os.path.exists(output_pt_path):
            raise FileNotFoundError(f"Expected output file not found: {output_pt_path}")

        return torch.load(output_pt_path, map_location=torch.device('cpu'))