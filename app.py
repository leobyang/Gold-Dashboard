import streamlit as st

from functions import load_main_gold
from tabs.gold_prices import render_gold_prices_tab
from tabs.macro_comparisons import render_macro_tab


def main():
    st.set_page_config(page_title="Gold Dashboard", layout="wide")
    st.title("Gold Dashboard")

    data = load_main_gold()

    tab1, tab2 = st.tabs(["Gold Prices", "Macro Comparisons"])

    with tab1:
        render_gold_prices_tab(data)

    with tab2:
        render_macro_tab(data)


if __name__ == "__main__":
    main()
