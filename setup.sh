mkdir -p ~/.streamlit/
pip install --upgrade pip
echo "\
[general]\n\
email = \"sebastian@portchain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml