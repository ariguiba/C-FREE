from typing import List
import json

from typing import List
import json
class Tokenizer():
    
    def __init__(self, path):
        self.path = path 
        #self.tokenizer = json.load(open(path))
        with open(path) as f:
            self.tokenizer = json.load(f)
        self.special_tokens = self.tokenizer['special_tokens']
        self.vocab_size = self.tokenizer['vocab_size']
        self.id_to_token = self.tokenizer.get('vocab', {})
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
    
    def tokenize(self, atom: List[int]):
        """
        tokenize an individual atom
        [x0, x1, ..., x8] becomes "x0|x1|...|x8"
        """
        return "|".join([str(prop) for prop in atom])
    
    def detokenize(self, atom: str):
        """
        method to convert from string atom "x0|x1|...|x8" back to list
        """
        if atom in self.special_tokens:
            return atom
        else:
            return [x for x in atom.split("|")]
    
    # def encode(self, atoms: List[List[int]]):
    #     """
    #     given a list of atoms, return a list of tokens
    #     each atom in atoms should be a list of 9 integers
    #     corresponding to the values of each of the 9 properties 
    #     """
    #     # convert each atom to a string first 
    #     tokens = [self.tokenize(atom) for atom in atoms]
    #     # return token ids 
    #     return [int(self.token_to_id.get(token, self.token_to_id.get('[UNK]', -1))) for token in tokens] 
    
    def encode(self, atoms: List[List[int]], mask: List[List[bool]]=None):
        """
        given a list of atoms, return a list of tokens
        each atom in atoms should be a list of 9 integers
        corresponding to the values of each of the 9 properties 
        """
        # convert each atom to a string first 
        if mask is None:
            tokens = [self.tokenize(atom) for atom in atoms]
        else:
            tokens = ['[MASK]' if m == 1 else self.tokenize(atom) for atom, m in zip(atoms, mask.squeeze().tolist())]
        # return token ids 
        return [int(self.token_to_id.get(token, self.token_to_id.get('[UNK]', -1))) for token in tokens]   
    

    def decode(self, tokens: List[List[int]]):
        """
        given a list of token ids, return a list of the original atoms 
        """
        atoms_string = [self.id_to_token.get(token) for token in tokens]
        # atoms in [x0, x1, ..., x8] form
        atoms = [self.detokenize(atom) for atom in atoms_string]
        return atoms
