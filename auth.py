import streamlit as st

from user_db import check_password, create_user, verify_user


def require_login():
    """Shows the Log in / Sign up page until the user is signed in. Returns the username."""
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]

    st.title("Explainable Sales Forecasting")
    st.caption("Log in or create an account to use the forecasting tool.")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Log in")
        if login_clicked:
            ok, result = verify_user(username, password)
            if ok:
                st.session_state["auth_user"] = result
                st.rerun()
            else:
                st.error(result)

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", help="3 to 30 characters: letters, numbers or underscore")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose a password", type="password", help="At least 8 characters, with a letter and a number")
            repeat_password = st.text_input("Repeat the password", type="password")
            signup_clicked = st.form_submit_button("Create account")
        if signup_clicked:
            if new_password != repeat_password:
                st.error("The two passwords do not match.")
            else:
                ok, message = create_user(new_username, new_email, new_password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
    st.stop()


def logout_button():
    def _logout():
        st.session_state.pop("auth_user", None)

    st.button("Log out", on_click=_logout)
