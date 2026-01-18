
from collections import Counter


class Ranking :

    def __init__(self,
                 query1,
                 query2,
                 retriever,
                 parent_data,
                 k=5,
                 ) -> None:
        
        self.query1 = query1
        self.query2 = query2
        self.k = k
        self.retriever = retriever
        self.parent_data = parent_data

    def parent_id_select(self,docs): 
        parent_ID = []
        duplicate_parents = []

        for i, doc in enumerate(docs, 1):  # start with 1 instead of 0
            parentid = doc.metadata["parent_id"]
            if parentid not in parent_ID :
                parent_ID.append(parentid)

            duplicate_parents.append(parentid)

        return parent_ID, duplicate_parents 


    def quering(self):
        docs1 = self.retriever.get_relevant_documents(self.query1)
        docs2 = self.retriever.get_relevant_documents(self.query2)

        parents1, duplicate_parents1 = self.parent_id_select(docs1)
        parents2, duplicate_parents2 = self.parent_id_select(docs2)

        duplicate_parents = duplicate_parents1 + duplicate_parents2

        parent_ID = parents1 + parents2

        """
        for i, doc in enumerate(docs, 1):  # start with 1 instead of 0
            parentid = doc.metadata["parent_id"]
            if parentid not in parent_ID :
                parent_ID.append(parentid)

            duplicate_parents.append(parentid)
        """



        counts = Counter(duplicate_parents)
        duplicates_sorted = {item: count for item, count in counts.items() if count > 1}

        try:
            top_parent = max(duplicates_sorted, key=duplicates_sorted.get)
        except:
            print("No top parent found")

        top_k = parent_ID[:self.k]

        try: 
            if top_parent not in top_k:
                top_k.append(top_parent)
        except:
            pass

        parent_chunk =[]

        for i in self.parent_data:
            if i.metadata["parent_id"] in top_k:
                parent_chunk.append(i.page_content)

        return (f"""Your given a question as such : {self.query1 + self.query2}\n , reply based on the data given below, 
            provide citation for each of your statement made based on the source given:\n \n {parent_chunk}""")