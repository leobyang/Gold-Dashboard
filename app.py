import streamlit as st

from functions import load_main_gold
from tabs.gold_prices import render_gold_prices_tab
from tabs.macro_comparisons import render_macro_tab
from tabs.compare_investments import render_compare_tab 
from tabs.scenario_analysis import render_scenario_tab  
from tabs.driver_sensitivity import render_driver_attribution_tab



def main():
    st.set_page_config(page_title="Gold Dashboard", layout="wide")
    st.title("Gold Dashboard")

    data = load_main_gold()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Gold Prices", "Macro Comparisons", "Compare Investments", "Driver Sensitivities", "Scenario Analysis"]
    )

    with tab1:
        render_gold_prices_tab(data)

    with tab2:
        render_macro_tab(data)

    with tab3:
        render_compare_tab(data)

    with tab4:
        render_driver_attribution_tab(data)

    with tab5:
        render_scenario_tab(data)
    
    
if __name__ == "__main__":
    main()
