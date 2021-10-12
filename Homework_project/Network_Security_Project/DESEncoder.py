import numpy as np
class Encoder:
    def __init__(self):
        '''
        Preparation for encoding, the size of ip and ipR are both 1*64
        
        '''
        self.sub_keys = []
        self.ip = np.array(
            [58,50,42,34,26,18,10,2,
            60,52,44,36,28,20,12,4,
            62,54,46,38,30,22,14,6,
            64,56,48,40,32,24,16,8,
            57,49,41,33,25,17,9,1,
            59,51,43,35,27,19,11,3,
            61,53,45,37,29,21,13,5,
            63,55,47,39,31,23,15,7
            ])
        self.ipR = np.array([
            40,8,48,16,56,24,64,32,
            39,7,47,15,55,23,63,31,
            38,6,46,14,54,22,62,30,
            37,5,45,13,53,21,61,29,
            36,4,44,12,52,20,60,28,
            35,3,43,11,51,19,59,27,
            34,2,42,10,50,18,58,26,
            33,1,41,9,49,17,57,25
        ])
        self.pc1 = np.array([
            57,49,41,33,25,17,9,
            1,58,50,42,34,26,18,
            10,2,59,51,43,35,27,
            19,11,3,60,52,44,36,
            63,55,47,39,31,23,15,
            7,62,54,46,38,30,22,
            14, 6,61,53,45,37,29,
            21,13,5,28,20,12,4
        ])
        self.pc2 = np.array([
            14,17,11,24,1,5,3,28,
            15,6,21,10,23,19,12,4,
            26,8,16,7,27,20,13,2,
            41,52,31,37,47,55,30,40,
            51,45,33,48,44,49,39,56,
            34,53,46,42,50,36,29,32
        ])
        self.round_text = []
        self.rotate = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]
        self.SBox = np.array([
            [
                14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7,
                0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8,
                4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0,
                15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13
            ],
            [
                15,1,8,14,6,11,3,4,9,7,2,13,12,0,5,10,
                3,13,4,7,15,2,8,14,12,0,1,10,6,9,11,5,
                0,14,7,11,10,4,13,1,5,8,12,6,9,3,2,15,
                13,8,10,1,3,15,4,2,11,6,7,12,0,5,14,9],
            [
                10,0,9,14,6,3,15,5,1,13,12,7,11,4,2,8,
                13,7,0,9,3,4,6,10,2,8,5,14,12,11,15,1,
                13,6,4,9,8,15,3,0,11,1,2,12,5,10,14,7,
                1,10,13,0,6,9,8,7,4,15,14,3,11,5,2,12     
            ],
            [
                7,13,14,3,0,6,9,10,1,2,8,5,11,12,4,15,
                13,8,11,5,6,15,0,3,4,7,2,12,1,10,14,9,
                10,6,9,0,12,11,7,13,15,1,3,14,5,2,8,4,
                3,15,0,6,10,1,13,8,9,4,5,11,12,7,2,14
            ],
            [
                2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9,
                14,11,2,12,4,7,13,1,5,0,15,10,3,9,8,6,
                4,2,1,11,10,13,7,8,15,9,12,5,6,3,0,14,
                11,8,12,7,1,14,2,13,6,15,0,9,10,4,5,3
            ],
            [
                12,11,0,15,9,2,6,8,0,13,3,4,14,7,5,11,
                10,15,4,2,7,12,9,5,6,1,13,14,0,11,3,8,
                9,14,15,5,2,8,12,3,7,0,4,10,1,13,11,6,
                4,3,2,12,9,5,15,10,11,14,1,7,6,0,8,13
            ],
            [
                4,11,2,14,15,0,8,13,3,12,9,7,5,10,6,1,
                13,0,11,7,4,9,1,10,14,3,5,12,2,15,8,6,
                1,4,11,13,12,3,7,14,10,15,6,8,0,5,9,2,
                6,11,13,8,1,4,10,7,9,5,0,15,14,2,3,12
            ],
            [
                13,2,8,4,6,15,11,1,10,9,3,14,5,0,12,7,
                1,15,13,8,10,3,7,4,12,5,6,11,0,14,9,2,
                7,11,4,1,9,12,14,2,0,6,10,13,15,3,5,8,
                2,1,14,7,4,10,8,13,15,12,9,0,3,5,6,11
            ]
        ],dtype="object")

    def rotateL(self,v,times):
        res = np.concatenate((v[times:],v[0:times]),axis=0)
        return res

    def hextobin(self,x):
        return bin(int(x, 16))[2:].zfill(64)

    def transform(self,bits,ip):
        '''
        Input: bits---key or text
               ip-- ip or pc vector 
        Output: res--the np array after transform

        '''
        if type(bits)==str: # Turn nstring into numpy array
            keylist = []
            for i in bits:
                keylist.append(int(i))
            bits = np.array(keylist)
        # print("ip: ",np.size(ip))
        res = []

        for i in ip:
            res.append(bits[i-1])
        # for i in range(np.size(ip,axis=0)):
        #     res[i] = bits[self.ip[i]-1]
        return np.array(res)

    def decompose(self,bits):
        part = len(bits)/2
        left = bits[:part]
        right = bits[part:]
        return left,right

    def key_gen(self,leftbits,rightbits):
        for round in range(16):
            leftone = self.rotateL(leftbits,self.rotate[round])
            rightone = self.rotateL(rightbits,self.rotate[round])
            combine_bits = np.concatenate((leftone,rightone),axis=0)
            leftbits = leftone
            rightbits = rightone
            subkey = self.transform(combine_bits,self.pc2)
            # for i in self.pc2:
            #     subkey.append(combine_bits[i - 1])
            self.sub_keys.append(subkey)
        return self.sub_keys
    
    def key_generation(self):
        key = input("enter key ")
        keybin = self.hextobin(key)
        key_length = len(keybin)
        if key_length < 64:
            while len(keybin) != 64:
                keybin = keybin+'0'
        keypc1 = self.transform(keybin,self.pc1)
        leftbits = keypc1[0:28]
        rightbits = keypc1[28:]
        self.key_gen(leftbits,rightbits)
        # for ele in self.sub_keys:
        #     print(ele)


    def expansion_box(self,plain_right):
        ep = [32,1,2,3,4,5,4,5,6,7,8,9,
     8,9,10,11,12,13,12,13,14,15,16,17,
     16,17,18,19,20,21,20,21,22,23,24,25,
     24,25,26,27,28,29,28,29,30,31,32,1]

        new_list = []
        for i in ep:
            new_list.append(plain_right[i-1])
        return new_list

    def check_xor_key(self,plain_exp,sub_key):
        plain_xor_list = []
        for i in range(0,len(plain_exp)):
            if plain_exp[i] == self.sub_keys[sub_key][i]:
                plain_xor_list.append(0)
            else:
                plain_xor_list.append(1)
        sub_key += 1
        return plain_xor_list



    def plain_substution(self,plain_xor):

        s_rows = ['00','01','10','11']

        s_cols = ['0000','0001','0010','0011','0100','0101','0110','0111','1000',
          '1001','1010','1011','1100','1101','1110','1111']


        plain_sub_list = []
        plain_list = []
        plain_xlist = []
        row = ""
        col = ""
        count = 0
        for i in range(0,len(plain_xor),6):
            value = plain_xor[i:i+6]
            plain_sub_list.append(value)
        for word in plain_sub_list:
            row = str(word[0])+str(word[-1])
            col = str(word[1])+str(word[2])+str(word[3])+str(word[4])
            box = self.SBox[count]
            # print(np.size(box,axis=0))
            val = box[s_rows.index(row)*16+s_cols.index(col)]
            plain_list.append(s_cols[val])

            count += 1
        for i in plain_list:
            for j in i:
                plain_xlist.append(int(j))
        return plain_xlist

    def plain_pbox(self,plain_sub):
        plain_p_text = []
        p = [16,7,20,21,29,12,28,17,1,15,23,26,5,18,31,10,
     2,8,24,14,32,27,3,9,19,13,30,6,22,11,4,25]
        for i in p:
            plain_p_text.append(plain_sub[i-1])
        return np.array(plain_p_text)

    def check_xor_left(self,plain_left,plain_P):
        plain_right_text = []
        for i in range(0,np.size(plain_left)):
            if plain_left[i] == plain_P[i]:
                plain_right_text.append(0)
            else:
                plain_right_text.append(1)

        return np.array(plain_right_text)

    def plain_ipR(self,round_plain_text):
        cipher_text_bin = []
        for i in self.ipR:
            cipher_text_bin.append(round_plain_text[i-1])
        return cipher_text_bin

    def cipher_text_conversion(self,cp_bin):
        cp_text = ""
        for i in cp_bin:
            cp_text += str(i)
        cipher_text = hex(int(cp_text,2))
        return cipher_text[2:]



    def encode(self):
        plain_text = input("enter plain text ")
        plain_text_bin = self.hextobin(plain_text)
        plain_text_length = len(plain_text_bin)
        if plain_text_length < 64:
            while len(plain_text_bin) != 64:
                plain_text_bin += '0'
        plain_ip = self.transform(plain_text_bin,self.ip)
        plain_left = plain_ip[0:32]
        plain_right = plain_ip[32:]
        rounds = 16
        sub_key = 0
        while rounds != 0:
            plain_exp = self.expansion_box(plain_right)
            plain_xor = self.check_xor_key(plain_exp,sub_key)
            plain_sub = self.plain_substution(plain_xor)
            plain_p = self.plain_pbox(plain_sub)
            right_part = self.check_xor_left(plain_left,plain_p)
            self.round_text.append(np.concatenate((plain_right,right_part),axis=0))
            plain_left = plain_right
            plain_right = right_part
            sub_key += 1
            rounds -= 1
        cipher_bin = self.plain_ip_1(self.round_text[15])
        cipher_text = self.cipher_text_conversion(cipher_bin)
        print("Cipher Text: ",cipher_text)

if __name__=="__main__":
    case = Encoder()
    case.key_generation()
    case.encode()