from openai import OpenAI
from config_setup import Config


setup = Config()

class llm: 
    def __init__ (
            self,
            user,
            system = setup.prompt ) :
        self.user = user
        self.system = system


    def llm_call(self):
        client = OpenAI(
            api_key = setup.api_key,
            base_url = "https://api.deepseek.com"
        )

        response = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [ {"role": "system", "content": self.system},
                         {"role": "user", "content":self.user}
                        
                    ]
        )
        output = response.choices[0].message.content
        return output

"""
if __name__ == "__main__":
        llm1 = llm("hello")
        output= llm1.llm_call()
        print(output.choices[0].message.content)

"""