import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

st.markdown(
    """
    <h1 styles= 'text-align: center; font-size: 50px;'>
    welcome to the streamlit
    <span style='color: red;'>hello</span>
    </h1> """,
    unsafe_allow_html=True
)
st.title(" this is the title of :blue[streamlit]")
st.header("this is the header of stream lit ", divider=True)
st.subheader(" this is the subheader of streamlit ",divider=True)
st.caption("this is the caption for streamlit used to define the caption in the heading or anything")
st.subheader(":red[code]",divider=True)
code = '''def hello():
            print("hello, streamlit")'''
st.code(code, language="python")
st.slider("this this the example of slider used in slider ",0 ,100,(10,70))
with st.echo():
    st.write("this is the text inside the echo ()")
    x=10
    st.write(f"the value of x is {x}")
tb= pd.DataFrame(np.random.randn(50,20),columns=('col %d' % i for i in range(20)))
st.dataframe(tb)
df= pd.DataFrame([
    {"command": "st.selectbox", "rating": 4, "is_widget": True},  #adding the data in the dataframe with command dictionary function;    
    {"command": "st.balloons", "rating": 5, "is_widget": False},
    {"command": "st.time_input", "rating": 3, "is_widget": True},
])
edited_df = st.data_editor(df)                          # this is used to edit the data in the dataframe
st.write(edited_df)                                     # this shows the edited data in the dataframe
                                                        # lets add allowance of row addition and deletion
edited_df = st.data_editor(df, num_rows="dynamic")      # this allows the user to add the row and delete the row dynamically

import pandas as pd
import streamlit as st

df = pd.DataFrame(
    [
        {"command": "st.selectbox", "rating": 4, "is_widget": True},
        {"command": "st.balloons", "rating": 5, "is_widget": False},
        {"command": "st.time_input", "rating": 3, "is_widget": True},
    ]
)
edited_df = st.data_editor(                     # using this create the data editor more interactive
    df,column_config={
        "command": "Streamlit Command",         # this is used to change the name of command column to Streamlit Command
        "rating": st.column_config.NumberColumn(# this is used to change the name of rating column to Your rating and also Configured as a number column with a range from 1 to 5, displayed with stars.
            "Your rating",
            help="How much do you like this command (1-5)?",
            min_value=1,
            max_value=5,
            step=1,
            format="%d ⭐",
        ),
        "is_widget": "Widget ?",
    },
    disabled=["command", "is_widget"],
    hide_index=True,
)

favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
st.markdown(f"Your favorite command is **{favorite_command}**")  #showing the favorite command in the markdown

# datetime column implementation
data_df = pd.DataFrame(
    {
        "appointment": [                                    #filling data in the dataframe with the tag of datetime  and values will be stored in the format of (yyyy,mm,dd,hh,mm).
            datetime(2024, 2, 5, 12, 30),
            datetime(2023, 11, 10, 18, 0),
            datetime(2024, 3, 11, 20, 10),
            datetime(2023, 9, 12, 3, 0),
        ]
    }
)

st.data_editor(
    data_df,
    column_config={
        "appointment": st.column_config.DatetimeColumn(             #using datetime coloumn config for ussing the datetime column in the dataframe and changed the name of appointment to Appointment
            "Appointment",
            min_value=datetime(2023, 6, 1),                         #setting the min value of the datetime column
            max_value=datetime(2025, 1, 1),
            format="D MMM YYYY, h:mm a",
            step=60,                                                 #step is used to set incrimentation in the code like her i used step of 60 so the time will be incremented by 60 minutes
        ),
    },
    hide_index=True,                                                 #used to hide the index of the dataframe
)
# implementation of only date column
data_df = pd.DataFrame(
    {
        "birthday": [
            date(1980, 1, 1),                      #filling the data in the dataframe with the tag of date and values will be stored in the format of (yyyy,mm,dd).
            date(1990, 5, 3),
            date(1974, 5, 19),
            date(2001, 8, 17),
        ]
    }
)

st.data_editor(
    data_df,
    column_config={
        "birthday": st.column_config.DateColumn(
            "Birthday",
            min_value=date(1900, 1, 1),             #setting the min value of the date
            max_value=date(2005, 1, 1),             #setting the max value of the date
            format="DD.MM.YYYY",                    #setting the format of the date
            step=1,                                #setting the step of date her is 1 cause the step is set to days here so the date will be incremented by 1 day
        ),
    },
    hide_index=True,
)
#impelementation of list column to show sales basic 
data_df = pd.DataFrame(
    {
        "sales": [
            ["apple", "banana", "cherry"],
            ["apple", "banana"],
            ["cherry"],
            ["apple", "cherry"],
        ]
    }
)
st.data_editor(
    data_df,
    column_config={
        "sales": st.column_config.ListColumn(         #using list column config to show the sales in the dataframe
            "Sales",
            help="Select the fruits that were sold",  #setting the help for the sales column
        )
    },
    hide_index=True,
)
#implementation of the link column
data_df = pd.DataFrame(
    {
        "apps": [
            "https://roadmap.streamlit.app",
            "https://extras.streamlit.app",
            "https://issues.streamlit.app",
            "https://30days.streamlit.app",
        ],
        "creator": [
            "https://github.com/streamlit",
            "https://github.com/arnaudmiribel",
            "https://github.com/streamlit",
            "https://github.com/streamlit",
        ],
    }
)

st.data_editor(
    data_df,
    column_config={
        "apps": st.column_config.LinkColumn(
            "Trending apps",
            help="The top trending Streamlit apps",
            validate=r"^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
            display_text=r"https://(.*?)\.streamlit\.app"
        ),
        "creator": st.column_config.LinkColumn(
            "App Creator", display_text="Open profile"
        ),
    },
    hide_index=True,
)
# we can also add dynamics in the link column so we can add additional rows and data in the in the out put like we did in selectbox column 
data = {
    "apps": [
        "https://app1.streamlit.app",
        "https://app2.streamlit.app",
        "https://app3.streamlit.app"
    ],
    "creator": [
        "https://profile1.streamlit.app",
        "https://profile2.streamlit.app",
        "https://profile3.streamlit.app"
    ]
}

# Create DataFrame
data_df = pd.DataFrame(data)

# Function to add a new row
def add_row(data_df):
    new_app = st.text_input("Enter new app URL:")
    new_creator = st.text_input("Enter new creator URL:")
    if st.button("Add"):
        if new_app and new_creator:
            new_row = pd.DataFrame({"apps": [new_app], "creator": [new_creator]})
            return pd.concat([data_df, new_row], ignore_index=True)
    return data_df

# Now add new row in the dataframe
data_df = add_row(data_df)

# Display the data in the dataframe
st.data_editor(
    data_df,
    column_config={
        "apps": st.column_config.LinkColumn(
            "Trending apps",
            help="The top trending Streamlit apps",
            validate=r"^https://[a-z]+\.streamlit\.app$",
            max_chars=100,
            display_text=r"https://(.*?)\.streamlit\.app"
        ),
        "creator": st.column_config.LinkColumn(
            "App Creator", display_text="Open profile"
        ),
    },
    hide_index=True,
    key="unique_data_editor_key"
)
data_df = pd.DataFrame(
    {
        "apps": [
            "https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/5435b8cb-6c6c-490b-9608-799b543655d3/Home_Page.png",
            "https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/ef9a7627-13f2-47e5-8f65-3f69bb38a5c2/Home_Page.png",
            "https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/31b99099-8eae-4ff8-aa89-042895ed3843/Home_Page.png",
            "https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/6a399b09-241e-4ae7-a31f-7640dc1d181e/Home_Page.png",
        ],
    }
)

st.data_editor(
    data_df,
    column_config={
        "apps": st.column_config.ImageColumn(
            "Preview Image", help="Streamlit app preview screenshots",width="medium",
        )
    },
    hide_index=True,
)
data_df = pd.DataFrame(
    {
        "sales": [
            [0, 4, 26, 80, 100, 40],
            [80, 20, 80, 35, 40, 100],
            [10, 20, 80, 80, 70, 0],
            [10, 100, 20, 100, 30, 100],
        ],
    }
)

st.data_editor(
    data_df,
    column_config={
        "sales": st.column_config.AreaChartColumn(
            "Sales (last 6 months)",
            width="medium",
            help="The sales volume in the last 6 months",
            y_min=0,
            y_max=100,
         ),
    },
    hide_index=True,
)
st.data_editor(
    data_df,
    column_config={
        "sales": st.column_config.LineChartColumn(
            "Sales (last 6 months)",
            width="medium",
            help="The sales volume in the last 6 months",
            y_min=0,
            y_max=100,
         ),
    },
    hide_index=True,
    key="unique_data_editor_key_for_line_chart"
)
st.data_editor(
    data_df,
    column_config={
        "sales": st.column_config.BarChartColumn(
            "Sales (last 6 months)",
            help="The sales volume in the last 6 months",
            y_min=0,
            y_max=100,
        ),
    },
    hide_index=True,
    key="unique_data_editor_key_for_bar_chart"
)
data_df = pd.DataFrame(
    {
        "sales": [2450, 545, 1045, 9089],
    }
)
st.data_editor(
    data_df,
    column_config={
        "sales": st.column_config.ProgressColumn(
            "Sales volume",
            help="The sales volume in USD",
            format="$%f",
            min_value=0,
            max_value=10000,
        ),
    },
    hide_index=True,
)
df1 = pd.DataFrame(
    np.random.randn(5,5), columns=("col %d" % i for i in range(5))
)

my_table = st.table(df1)

df2 = pd.DataFrame(
    np.random.randn(5,5), columns=("col %d" % i for i in range(5))
)

my_table.add_rows(df2)
my_chart = st.line_chart(df1)
my_chart.add_rows(df2)

my_chart = st.vega_lite_chart(
    {
        "mark": "line",
        "encoding": {"x": "a", "y": "b"},
        "datasets": {
            "example": df1, 
        },
        "data": {"name": "example"},
    }
)
my_chart.add_rows(example=df2)