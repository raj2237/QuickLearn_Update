from pinecone import Pinecone
pc = Pinecone(api_key="pcsk_4TTRAg_GXrXt8159BC4qcfQCPkTtQRmLKJUhiBoT8h7VYB9bBtSp9XFnPLGRQoPaiUwnJf")
print(pc.list_indexes().names())