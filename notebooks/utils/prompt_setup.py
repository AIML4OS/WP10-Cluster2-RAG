# Functions for building prompts
import pnadas as pd

intro_prompt = "You are an expert in business industry classification. Your task is to assign the single most appropriate Norwegian NACE 5-digit (subclass) revision 2.1 industry code to a company based on the information provided."


def build_prompt_standard(
    company_name: str,
    company_activity: str,
    company_purpose: str,
    hits: list[dict],
) -> str:
    
    candidates = "\n".join(
        f"{i+1}. {h['code']}: {h['description']}"
        for i, h in enumerate(hits)
    )

    prompt = f"""{intro_prompt}

    ## Company information
    - **Name**: {company_name}
    - **Activity**: {company_activity}
    - **Purpose**: {company_purpose}
    
    ## Candidate codes
    The following industry codes have been retrieved as the most semantically similar to the company description:
    
    {candidates}
    
    ## Task
    Select the single best matching code from the candidates above. If none of the candidates fit, choose the best one.
    
    Respond with only the code and nothing else.
    """
    return prompt

def build_prompt_zero(
    company_name: str,
    company_activity: str,
    company_purpose: str,
) -> str:

    prompt = f"""{intro_prompt}

    ## Company information
    - **Name**: {company_name}
    - **Activity**: {company_activity}
    - **Purpose**: {company_purpose}
    
    ## Task
    Select the single best matching Norwegian NACE revision 2.1 5-digit code for the company. If none of the candidates fit, choose the best one.
    
    Respond with only the code and nothing else.
    """
    return prompt

def build_prompt_descriptions(
    company_name: str,
    company_activity: str,
    company_purpose: str,
    sn_descriptions,
) -> str:

    prompt = f"""{intro_prompt}

    ## Company information
    - **Name**: {company_name}
    - **Activity**: {company_activity}
    - **Purpose**: {company_purpose}

    ## Class descriptions
    {sn_descriptions}
    
    ## Task
    Using the class desriptions as guidance, select the single best matching Norwegian NACE revision 2.1 5-digit code for the company. If none of the candidates fit, choose the best one.
    
    Respond with only the code and nothing else.
    """
    return prompt

def build_prompt_obs(
    company_name: str,
    company_activity: str,
    company_purpose: str,
    hits: list[dict],
) -> str:
    
    prompt = f"""{intro_prompt}

    ## Company information
    - **Name**: {company_name}
    - **Activity**: {company_activity}
    - **Purpose**: {company_purpose}
    
    ## Similar labelled examples
    The following are real companies from our training data that are semantically similar to the company you are classifying:

    {format_few_shot_examples(hits)}
    
    ## Task
    Using the labelled examples as guidance, select the single best matching NACE revision 2.1 code for the company. If none of the candidates fit, choose the best one.
    
    Respond with only the code and nothing else.
    """
    return prompt

    
def build_prompt_fasttext(
    company_name: str,
    company_activity: str,
    company_purpose: str,
    hits: list[dict],
    sn: pd.DataFrame,
) -> str:
    
    candidates = format_candidates(hits, sn)

    prompt = f"""{intro_prompt}

    ## Company information
    - **Name**: {company_name}
    - **Activity**: {company_activity}
    - **Purpose**: {company_purpose}
    
    ## Candidate codes
    The following subclasses and descriptions been retrieved as the most semantically similar to the company description:
    
    {candidates}
    
    ## Task
    Select the single best matching Norwegian NACE revision 2.1 5-digit code for the company. If none of the candidates fit, choose the best one.
    
    Respond with only the code and nothing else.
    """
    return prompt

def format_candidates(hits: list, sn: pd.DataFrame, notes = True):
    candidates = ""
    for h in hits:
        s = sn.loc[sn.code == h, "name"]
        sn_name = s.iloc[0] if not s.empty else ""
        n = sn.loc[sn.code == h, "notes"] 
        sn_notes = n.iloc[0] if not n.empty else ""
        if notes:
            newline = f"{h}: {sn_name} {sn_notes}".strip()
        else:
            newline = f"{h}: {sn_name}".strip()
        candidates = f"{candidates}\n {newline}"
    return candidates