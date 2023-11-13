import time
import random
import requests
import scrapy
import os

headers = {
  "User-agent":
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 "
  "Edge/18.19582"
}


def simple_google_search(query, retries=2):
    params = {
      "q": query,
    }
    html = requests.get('https://www.google.com/search', headers=headers, params=params)
    # read the html content with scrapy
    content = scrapy.Selector(text=html.content)
    response = ""
    try:
        calc = get_calculator(content)
        response += f"Expression: {query}\nAnswer: {calc}\n"
    except:
        pass
    try:
        weather = get_weather(content)
        return weather
    except:
        pass
    try:
        population = get_population(content)
        if population.find(r'[0-9]') > -1:
            return f"Population: {population}\n"

    except:
        pass
    try:
        second_answer = get_second_answer(content)
        if len(second_answer) > 2 and second_answer[:3] != "Ad\n":
            return f'Answer: {second_answer}\n'
    except:
        pass
    try:
        answer = get_answer(content)
        if len(answer) > 2 and answer[:3] != "Ad\n":
            return f'Answer: {answer}\n'
    except:
        pass
    try:
        basic = get_answer_basic(content)
        if len(basic) > 2 and basic[:3] != "Ad\n":
            return f'Answer: {basic}\n'
    except:
        pass
    try:
        address = get_address(content)
        if len(address) > 2:
            return address
    except:
        pass
    try:
        conversion = get_conversion(content)
        if len(conversion) > 2:
            return conversion
    except:
        pass
    try:
        translation = get_translation(content)
        if len(translation) > 2:
           return f"Translation: {translation}\n"
    except:
        pass
    try:
        flights_one = get_flights_one(content)
        if flights_one.find(r'[0-9]') > -1:
            return f"Flights: {flights_one}\n"
    except:
        pass
    try:
        flights_two = get_flights_two(content)
        if flights_two.find(r'[0-9]') > -1:
            return f"Flights: {flights_two}\n"
    except:
        pass
    try:
        sports = get_sports(content)
        if sports.find(r'[0-9]') > -1:
            return f"Sports: {sports}\n"
    except:
        pass
    if retries > 0:
        time.sleep(4+random.random()*2)
        return simple_google_search(query, retries=retries-1)
    return "No simple answer found."


def test_google_search():
    tests = ["Address of the Fox Building", "2*2+2", "Baltimore weather", "London population", "100 usd in gbp",
             "swagger definition", "luke skywalker lightsaber color", "Who is using IBM", "hello in french",
             "Flights from Baltimore to Miami", "Flights from Baltimore to Miami on Tuesday", "nfl matches"]
    tests = random.sample(tests, len(tests))
    fails = []
    errors = []
    for query in tests:
        try:
            result = simple_google_search(query)
            time.sleep(4+random.random()*2)
            if result == "No simple answer found.":
                fails.append(query)
                print(f"Test failed: {query}")
            else:
                print(f"Test passed: {query}:{result}")
        except Exception as e:
            errors.append(query)
            print(f"Test error: {query} - {e}")
    if len(fails) == 0 and len(errors) == 0:
        print("All tests passed!")
        return
    else:
        raise Exception("simple_google_search tests failed.")


def open_search_tool(query):
    params = {
        "q": query,
    }
    html = requests.get('https://www.google.com/search', headers=headers, params=params)
    # read the html content with scrapy
    content = scrapy.Selector(text=html.content)
    with open("test.html", "w") as f:
        f.write(html.text)
    # open the file in a browser
    os.system("open test.html")
    return content


# Address of the Fox Building
def get_address(content):
    address = content.xpath('//*[@id="main"]/div[3]/div/div[3]/div/div/div/div/div/div/div/div/div/text()')[0].get()
    return address


# 2*2+2
def get_calculator(content):
    calc = content.xpath('//*[@id="main"]/div[3]/div/div[3]/div/div/div/div/div/div/div/div/div/text()')[0].get()
    return calc


# Baltimore weather
def get_weather(content):
    temperature = content.xpath('//*[@id="main"]/div[3]/div/div[3]/div/div/div/div/div[1]/div[1]/div/div/div/div/text()')[0].get()
    condition = content.xpath('//*[@id="main"]/div[3]/div/div[3]/div/div/div/div/div[1]/div[2]/div/div/div/div/text()')[0].get()
    condition = condition.replace("\u202f", "")
    return temperature + "\n" + condition


# London population
def get_population(content):
    population = content.xpath('//*[@id="main"]/div[3]/div/div[3]/div/div/div/div/div[1]/div[1]/div/div/div/div/text()')[0].get()
    return population


# 100 usd in gbp
def get_conversion(content):
    conversion = content.xpath('//*[@id="main"]/div[3]/div/div[3]/div/div/div/div/div[1]/div/div/div/div/text()')[0].get()
    return conversion


# swagger definition
def get_answer_basic(content):
    definition = "\n".join(content.xpath('//*[@id="main"]/div[3]/div/div[3]/div/div/*//text()').extract())
    return definition


# luke skywalker lightsaber color
def get_answer(content):
    answer = "\n".join(content.xpath('//*[@id="main"]/div[3]/div/div[2]/div/*//text()').extract())
    return answer


# Who is using IBM
def get_second_answer(content):
    answer = "\n".join(content.xpath('//*[@id="main"]/div[3]/div/div/*//text()').extract())
    return answer


# hello in french
def get_translation(content):
    translation_input_language = content.xpath('//*[@id="tsuid_2"]/option[contains(@selected, "selected")]/text()').get()
    translation_input = content.xpath('//*[@id="lrtl-source-text"]/input/@value').get()
    translation_output_language = content.xpath('//*[@id="tsuid_4"]/option[contains(@selected, "selected")]/text()').get()
    translation_output = content.xpath('//*[@id="lrtl-translation-text"]/text()').get()
    if translation_input_language is not None and translation_input is not None and \
        translation_output_language is not None and translation_output is not None:
        translation = f"{translation_input_language}: {translation_input}\n{translation_output_language}: {translation_output}"
        return translation
    else:
        return ""


# Flights from Baltimore to Miami
def get_flights_one(content):
    flights = "\n".join(content.xpath('//*[@id="main"]/div[4]/div/div/div[1]/*//text()').extract())
    return flights


# Flights from Baltimore to Miami on Tuesday
def get_flights_two(content):
    flights = "\n".join(content.xpath('//*[@id="main"]/div[3]/*//text()').extract())
    return flights


# nfl matches
def get_sports(content):
    sports = "\n".join(content.xpath('//*[@id="main"]/div[3]/div*//text()').extract())
    return sports