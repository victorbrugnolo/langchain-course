from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

long_text = """
# Spider-Man vs Green Goblin: The Ultimate Rivalry

The conflict between Spider-Man and the Green Goblin stands as one of the most iconic and personal rivalries in comic book history. This isn't just a battle between hero and villain—it's a deeply tragic story of friendship, betrayal, madness, and loss that has defined Spider-Man's character for decades.

## The Origins of the Green Goblin

Norman Osborn, a brilliant industrialist and founder of Oscorp, became the Green Goblin through a combination of ambition and accident. Seeking to create a formula that would enhance human strength and intelligence, Osborn experimented on himself with a serum that granted him superhuman abilities but also drove him to madness. The formula amplified his worst traits: his ruthlessness, his need for control, and his capacity for cruelty.

As the Green Goblin, Osborn donned a terrifying Halloween-inspired costume, complete with a grotesque goblin mask, and armed himself with an arsenal of deadly weapons. His glider became his signature mode of transport, while his pumpkin bombs and razor bats made him one of Spider-Man's most dangerous foes. But what truly set the Goblin apart was his intelligence—he was a tactical genius who could match Spider-Man's wit while surpassing him in resources and ruthlessness.

## The Personal Connection

What makes the Spider-Man vs Green Goblin conflict so devastating is the personal connection between them. Norman Osborn was the father of Harry Osborn, Peter Parker's best friend. Peter looked up to Norman as a father figure, especially after losing his own Uncle Ben. This relationship made their eventual conflict all the more tragic—Peter wasn't just fighting a supervillain, he was fighting the father of his closest friend.

Norman, in his moments of lucidity, genuinely cared for Peter. But when the Goblin personality took over, that affection twisted into obsession and hatred. The Goblin saw Spider-Man as both a worthy adversary and an obstacle to be destroyed. This duality created a psychological warfare that went far beyond physical confrontation.

## The Death of Gwen Stacy

The rivalry reached its darkest moment in one of comic book history's most infamous storylines. The Green Goblin discovered Spider-Man's secret identity and used this knowledge to strike at Peter's most vulnerable point—the people he loved. In a moment that would haunt Spider-Man forever, the Goblin kidnapped Gwen Stacy, Peter's girlfriend, and threw her from the George Washington Bridge.

Spider-Man shot out a web to catch her, but the sudden stop broke her neck, killing her instantly. Whether it was the fall or the webbing that killed her became a source of eternal debate and guilt for Peter. This single act transformed their rivalry from a typical hero-villain dynamic into something far more personal and unforgivable. The Green Goblin had taken something from Spider-Man that could never be replaced.

In the same battle, Norman Osborn appeared to die when his own glider impaled him—a remote control malfunction as he tried to kill Spider-Man from behind. But death would not be permanent for the Green Goblin, and his return would continue to plague Spider-Man's life.

## The Psychological Battle

The Green Goblin's greatest weapon against Spider-Man has always been psychological torment. Unlike villains who simply want to rob banks or take over the world, the Goblin wants to break Spider-Man's spirit. He wants to prove that beneath the heroic exterior, Peter Parker is just as capable of darkness and violence as any villain.

The Goblin has repeatedly attacked Peter's loved ones, knowing that Spider-Man's greatest strength—his compassion and sense of responsibility—is also his greatest weakness. Every punch Spider-Man throws at the Goblin carries the weight of guilt, anger, and the knowledge that this is his best friend's father. Every time Peter holds back, the Goblin exploits that mercy.

Norman has tried to turn Peter to his side, offering him power, wealth, and freedom from responsibility. He's tempted Peter with the idea that they're not so different—two geniuses burdened by the expectations of others. But Peter always rejects these overtures, knowing that the path the Goblin offers leads only to destruction and moral emptiness.

## The Legacy Continues

The Spider-Man vs Green Goblin conflict didn't end with Norman. His son Harry eventually discovered his father's identity and, consumed by grief and anger over Norman's death (which he blamed on Spider-Man), became the second Green Goblin. This forced Peter to fight his best friend, adding another layer of tragedy to an already painful legacy.

The Goblin legacy has passed through multiple generations, with various individuals taking up the mantle. But Norman Osborn remains the definitive Green Goblin, the one whose intelligence, resources, and sheer malevolence make him Spider-Man's most dangerous enemy.

## Why This Rivalry Endures

The Spider-Man vs Green Goblin conflict resonates because it represents the cost of being a hero. Peter Parker could have walked away from being Spider-Man, could have prioritized his own happiness and safety, but his sense of responsibility wouldn't allow it. The Green Goblin punishes him for that choice, repeatedly proving that heroism comes with terrible sacrifice.

Their battles raise profound questions about identity, responsibility, and the nature of evil. Is Norman Osborn responsible for the crimes of the Green Goblin, or is the Goblin a separate entity created by the formula? Can someone who has caused so much pain ever be redeemed? And how much is Spider-Man willing to sacrifice in his mission to protect others?

The physical battles between Spider-Man and the Green Goblin are spectacular—high-speed chases through New York City, explosive confrontations, and desperate struggles on top of bridges and buildings. But the real battle is psychological and moral. It's about a young man trying to do the right thing in a world that constantly punishes him for it, and a villain who represents everything that could go wrong when brilliance is corrupted by ego and madness.

## The Modern Era

In contemporary comics and films, the Spider-Man vs Green Goblin rivalry has been reimagined for new audiences while maintaining its core elements. The Sam Raimi Spider-Man films brought this conflict to mainstream audiences with Willem Dafoe's chilling portrayal of Norman Osborn. The films captured the tragedy of their relationship and the horror of the Green Goblin's actions, introducing a new generation to this classic rivalry.

More recently, the Marvel Cinematic Universe has begun exploring its own version of these characters, while the comics continue to find new ways to deepen and complicate their relationship. Norman Osborn has even led major villain organizations and temporarily positioned himself as a hero during the "Dark Reign" storyline, showing that the threat he represents extends far beyond just Spider-Man.

## Conclusion

The rivalry between Spider-Man and the Green Goblin transcends typical superhero conflicts. It's a story about the price of heroism, the tragedy of mental illness, the burden of secrets, and the way violence cascades through generations. It's about a young man who tries to save everyone and a villain who exploits that noble impulse to cause maximum pain.

Every punch thrown, every web slung, and every confrontation between these two characters carries decades of history, pain, and unresolved tragedy. The Green Goblin isn't just Spider-Man's enemy—he's the dark mirror that shows Peter Parker what he could become if he ever abandoned his principles. And Spider-Man isn't just the Goblin's opponent—he's the reminder of everything Norman Osborn could have been if he hadn't let ambition and madness consume him.

This is why, after more than sixty years, the conflict between Spider-Man and the Green Goblin remains one of the most compelling and emotionally resonant rivalries in all of fiction.
"""

spliter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=70,
)

chunks = spliter.create_documents([long_text])

# for chunck in chunks:
#     print(chunck.page_content)
#     print("----"*30)

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

map_prompt = PromptTemplate.from_template("Write a concise summary of the following text:\n{context}")
map_chain = map_prompt | llm | StrOutputParser()

prepare_map_inputs = RunnableLambda(lambda docs: [{"context": doc.page_content} for doc in docs])
map_stage = prepare_map_inputs | map_chain.map()

reduce_prompt = PromptTemplate.from_template("Combine the following summaries into a single concise summary:\n{context}")
reduce_chain = reduce_prompt | llm | StrOutputParser()

prepare_reduce_inputs = RunnableLambda(lambda summaries: {"context": "\n".join(summaries)})
pipeline = map_stage | prepare_reduce_inputs | reduce_chain

result = pipeline.invoke(chunks)
print(result)
