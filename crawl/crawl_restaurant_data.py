import json
import re
import logging
import random
import threading
import queue
import time
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import quote
import os

@dataclass
class CrawlerConfig:
    base_url: str = "https://map.naver.com/p/search/"
    default_wait_time: float = 1  # 
    max_review_clicks: int = 10
    scroll_wait_time: float = 1  # 스크롤 대기 시간
    click_wait_time: float = 1   # 클릭 후 대기 시간
    max_workers: int = 3  # 병렬 처리 워커 수 (사용자 요청)
    
    # CSS Selectors
    search_iframe_id: str = "searchIframe"
    entry_iframe_id: str = "entryIframe"
    first_result_selector: str = "li.UEzoS:first-child .place_bluelink"
    tab_container_selector: str = ".Jxtsc"
    menu_more_button_selector: str = ".fvwqf"
    # Multiple menu item selectors for different layouts
    menu_item_selectors: list[str] = None
    menu_name_selectors: list[str] = None
    menu_price_selectors: list[str] = None
    # Sub-tab selectors (for menu category tabs etc.)
    review_more_button_selector: str = "a.fvwqf"
    review_container_selector: str = "ul#_review_list"
    review_item_selector: str = "div.pui__vn15t2 > a"
    # Restaurant info selectors
    info_description_selectors: list[str] = None
    
    def __post_init__(self):
        # Define multiple selector patterns for different menu layouts
        self.menu_item_selectors = [
            "div.MenuContent__info_detail__rCviz",  # For Starbucks-like infinite scroll menus
            "div.MXkFw",  # Original selector for standard layout
        ]
        self.menu_name_selectors = [
            "div.MenuContent__tit__313LA", # For Starbucks-like infinite scroll menus
            ".lPzHi",  # Original selector
        ]
        self.menu_price_selectors = [
            "div.MenuContent__price__lhCy9", # For Starbucks-like infinite scroll menus
            ".GXS1X",  # Original selector
        ]
        self.info_description_selectors = [
            "div.T8RFa",  # New requirement: specific selector for description
        ]

# 설정 및 경로
config = CrawlerConfig()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "../data/restaurants.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "../data/crawled_restaurants")
FAILED_QUERIES_FILE = os.path.join(BASE_DIR, "../data/crawl_failed_queries.txt")
MAX_RECORDS_PER_FILE = 1000

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# 파일 쓰기 및 공유 데이터 접근을 위한 전역 락
file_write_lock = threading.Lock()
shared_data_lock = threading.Lock()


def get_driver() -> webdriver.Chrome:
    """WebDriver를 설정하고 반환합니다."""
    logger.info("WebDriver 초기화 중...")
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    options.add_argument("--start-maximized")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
    }
    options.add_experimental_option("prefs", prefs)
    
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(5)
    logger.info("WebDriver 초기화 완료")
    return driver


def wait_for_element_clickable(driver: webdriver.Chrome, selector: str, timeout: int = 5) -> bool:
    """요소가 클릭 가능할 때까지 효율적으로 대기합니다."""
    try:
        WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
        return True
    except TimeoutException:
        return False


def switch_to_entry_iframe(driver: webdriver.Chrome, wait: WebDriverWait) -> bool:
    try:
        driver.switch_to.default_content()
        wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, config.entry_iframe_id)))
        logger.info("단일 상세 페이지 Iframe으로 전환 성공")
        return True
    except TimeoutException:
        return False


def click_tab(driver: webdriver.Chrome, wait: WebDriverWait, tab_name: str) -> bool:
    """지정된 탭을 클릭합니다."""
    try:
        try:
            tab_container = driver.find_element(By.CSS_SELECTOR, ".Jxtsc")
            all_tabs = tab_container.find_elements(By.TAG_NAME, "a")
            available_tabs = [tab.text.strip() for tab in all_tabs if tab.text.strip()]
            logger.info(f"사용 가능한 탭들: {available_tabs}")
        except:
            logger.warning("탭 목록을 가져올 수 없습니다.")
        
        tab_patterns = [
            f"//div[contains(@class, 'Jxtsc')]//a[.//span[text()='{tab_name}']]",
            f"//a[.//span[text()='{tab_name}']]",
        ]
        
        for pattern in tab_patterns:
            try:
                tab = wait.until(EC.element_to_be_clickable((By.XPATH, pattern)))
                logger.info(f"'{tab_name}' 탭을 찾았습니다. pattern: {pattern}")
                driver.execute_script("arguments[0].click();", tab)
                WebDriverWait(driver, config.default_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
                )
                return True
            except TimeoutException:
                continue
        
        logger.warning(f"'{tab_name}' 탭을 찾을 수 없습니다.")
        return False
    except Exception as e:
        logger.error(f"탭 클릭 중 오류 발생: {e}")
        return False


def click_more_button_until_done(driver: webdriver.Chrome, selector: str, max_clicks: int = None) -> int:
    """더보기 버튼을 찾을 수 없을 때까지 계속 클릭합니다."""
    click_count = 0
    max_clicks = max_clicks or float('inf')
    
    more_button_selectors = [selector, "a.fvwqf"]
    
    while click_count < max_clicks:
        button_found = False
        for btn_selector in more_button_selectors:
            try:
                more_button = WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, btn_selector))
                )
                driver.execute_script("arguments[0].click();", more_button)
                click_count += 1
                button_found = True
                time.sleep(config.click_wait_time)
                break
            except TimeoutException:
                continue
            except Exception as e:
                logger.debug(f"더보기 버튼 클릭 중 예외 ({btn_selector}): {e}")
                continue
        
        if not button_found:
            break
    
    if click_count > 0:
        logger.info(f"총 {click_count}번의 더보기 버튼을 클릭했습니다.")
    return click_count


def validate_menu_item(name: str, price: str) -> bool:
    """메뉴 아이템의 유효성을 검증합니다."""
    if not (name and name.strip() and price and price.strip()):
        return False
    if name.strip() == price.strip():
        return False
    return True


def try_menu_selectors(driver: webdriver.Chrome, wait: WebDriverWait) -> tuple | None:
    """다양한 CSS 셀렉터를 시도하여 메뉴 항목을 찾습니다."""
    for item_selector in config.menu_item_selectors:
        try:
            menu_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, item_selector)))
            if menu_elements:
                for name_selector in config.menu_name_selectors:
                    for price_selector in config.menu_price_selectors:
                        try:
                            test_el = menu_elements[0]
                            name = test_el.find_element(By.CSS_SELECTOR, name_selector).text
                            price = test_el.find_element(By.CSS_SELECTOR, price_selector).text
                            if validate_menu_item(name, price):
                                logger.info(f"메뉴 셀렉터 찾음: items={item_selector}, name={name_selector}, price={price_selector}")
                                return (menu_elements, name_selector, price_selector)
                        except (NoSuchElementException, TimeoutException):
                            continue
        except TimeoutException:
            continue
    return None


def try_price_based_extraction(driver: webdriver.Chrome) -> list[dict[str, str]]:
    """가격 기반으로 메뉴 항목을 추출합니다. (스타벅스 등 특수 구조용)"""
    try:
        price_patterns = [
            "//*[contains(text(), '원')]", "//*[contains(text(), '₩')]",
            "//*[contains(text(), ',')]", "//*[text()[matches(., '\d{3,}')]]"
        ]
        all_price_elements = []
        for pattern in price_patterns:
            try:
                all_price_elements.extend(driver.find_elements(By.XPATH, pattern))
            except:
                continue
        
        valid_menu_items = []
        processed_prices = set()
        processed_names = set()
        
        for elem in all_price_elements:
            try:
                price_text = elem.text.strip()
                price_re_patterns = [r'^\d{1,2},?\d{3}원$', r'^\d{4,5}원$', r'^\d{1,2},?\d{3}$', r'^₩\d{1,2},?\d{3}$']
                if not any(re.match(p, price_text) for p in price_re_patterns) or price_text in processed_prices:
                    continue
                processed_prices.add(price_text)
                
                current = elem
                found_menu = False
                for _ in range(8):
                    try:
                        current = current.find_element(By.XPATH, "..")
                        search_elements = current.find_elements(By.XPATH, ".//*") + current.find_elements(By.XPATH, "../../*")
                        
                        for search_elem in search_elements:
                            sibling_text = search_elem.text.strip()
                            if (sibling_text and price_text not in sibling_text and 2 <= len(sibling_text) <= 150 and
                                '원' not in sibling_text and not re.search(r'\d{3,}', sibling_text) and
                                sibling_text not in processed_names):
                                
                                category_keywords = ['음료', '추천', '커피', '블렌디드', '티', '메뉴', '카테고리', '리뷰']
                                if not any(keyword in sibling_text for keyword in category_keywords):
                                    valid_menu_items.append({"name": sibling_text, "price": price_text})
                                    processed_names.add(sibling_text)
                                    found_menu = True
                                    break
                        if found_menu: break
                    except:
                        break
            except:
                continue
        
        unique_menus = [dict(t) for t in {tuple(d.items()) for d in valid_menu_items}]
        logger.info(f"가격 기반 추출로 {len(unique_menus)}개 메뉴 발견")
        return unique_menus
    except Exception as e:
        logger.warning(f"가격 기반 추출 중 오류: {e}")
        return []


def scroll_until_no_new_content(driver: webdriver.Chrome, wait: WebDriverWait, item_selector: str) -> int:
    """무한 스크롤 페이지에서 새로운 콘텐츠가 없을 때까지 스크롤합니다."""
    scroll_count, no_new_content_strikes = 0, 0
    while no_new_content_strikes < 3:
        try:
            menu_elements = driver.find_elements(By.CSS_SELECTOR, item_selector)
            initial_item_count = len(menu_elements)
            if not menu_elements: break

            driver.execute_script("arguments[0].scrollIntoView(true);", menu_elements[-1])
            scroll_count += 1
            time.sleep(config.scroll_wait_time)

            final_item_count = len(driver.find_elements(By.CSS_SELECTOR, item_selector))
            if final_item_count > initial_item_count:
                no_new_content_strikes = 0
            else:
                no_new_content_strikes += 1
        except Exception as e:
            logger.warning(f"스크롤 중 예외 발생: {e}")
            break
    logger.info(f"{scroll_count}번 스크롤 후 종료.")
    return scroll_count


def get_menu_data(driver: webdriver.Chrome, wait: WebDriverWait) -> list[dict[str, str]] | None:
    """메뉴 탭의 모든 메뉴명과 가격을 수집합니다."""
    if not click_tab(driver, wait, "메뉴"): return None
    logger.info("메뉴 정보 수집 시작")

    click_more_button_until_done(driver, config.menu_more_button_selector, max_clicks=20)
    time.sleep(config.click_wait_time)

    try:
        infinite_scroll_selector = "div.MenuContent__info_detail__rCviz"
        if driver.find_elements(By.CSS_SELECTOR, infinite_scroll_selector):
            scroll_until_no_new_content(driver, wait, infinite_scroll_selector)
            time.sleep(config.scroll_wait_time)
    except Exception as e:
        logger.warning(f"무한 스크롤 확인 중 오류 발생: {e}")

    menus = []
    menu_result = try_menu_selectors(driver, wait)
    if menu_result:
        menu_elements, name_selector, price_selector = menu_result
        for el in menu_elements:
            try:
                name = el.find_element(By.CSS_SELECTOR, name_selector).text
                price = el.find_element(By.CSS_SELECTOR, price_selector).text
                if validate_menu_item(name, price):
                    menus.append({"name": name.strip(), "price": price.strip()})
            except NoSuchElementException:
                continue
    else:
        menus = try_price_based_extraction(driver)
    
    if not menus: return None
    
    unique_menus = [dict(t) for t in {tuple(d.items()) for d in menus}]
    logger.info(f"총 {len(unique_menus)}개의 유효한 메뉴를 수집했습니다.")
    return unique_menus


def validate_review_text(text: str) -> bool:
    """리뷰 텍스트의 유효성을 검증합니다."""
    return bool(text and text.strip() and len(text.strip()) > 5)


def get_info_description(driver: webdriver.Chrome, wait: WebDriverWait) -> str | None:
    """정보 탭의 업체 소개를 수집합니다."""
    for tab_name in ["정보", "Info", "상세정보"]:
        if click_tab(driver, wait, tab_name):
            logger.info("업체 소개 정보 수집 시작")
            for selector in config.info_description_selectors:
                try:
                    element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                    return element.text.strip()
                except TimeoutException:
                    continue
            logger.warning("유효한 업체 소개를 찾지 못했습니다.")
            return None
    logger.warning("정보 탭을 찾을 수 없습니다.")
    return None


def get_review_data(driver: webdriver.Chrome, wait: WebDriverWait) -> list[str]:
    """리뷰 탭의 모든 텍스트 리뷰를 수집합니다."""
    for tab_name in ["리뷰", "후기", "Review"]:
        if click_tab(driver, wait, tab_name):
            logger.info("리뷰 정보 수집 시작")
            click_more_button_until_done(driver, config.review_more_button_selector, config.max_review_clicks)
            
            reviews = []
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.review_container_selector)))
                review_items = driver.find_elements(By.CSS_SELECTOR, f"{config.review_container_selector} > li")
                for item in review_items:
                    try:
                        review_text = item.find_element(By.CSS_SELECTOR, config.review_item_selector).text.strip()
                        if validate_review_text(review_text):
                            reviews.append(review_text)
                    except NoSuchElementException:
                        continue
                logger.info(f"총 {len(reviews)}개의 유효한 리뷰를 수집했습니다.")
                return reviews
            except TimeoutException:
                logger.error(f"'{config.review_container_selector}' 컨테이너를 찾지 못했습니다.")
            return []
    logger.warning("리뷰 탭을 찾을 수 없습니다.")
    return []


def load_existing_crawled_data(output_dir: str) -> tuple[set, dict]:
    """기존에 크롤링된 업체 ID와 검색 키워드 매핑을 로드합니다."""
    crawled_place_ids, search_keyword_to_place_id = set(), {}
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith('part-') and filename.endswith('.jsonl'):
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'place_id' in data:
                                crawled_place_ids.add(data['place_id'])
                                if 'search_keyword' in data:
                                    search_keyword_to_place_id[data['search_keyword']] = data['place_id']
                        except json.JSONDecodeError:
                            continue
    logger.info(f"기존 크롤링 데이터 로드: 업체 ID {len(crawled_place_ids)}개, 검색 키워드 {len(search_keyword_to_place_id)}개")
    return crawled_place_ids, search_keyword_to_place_id


def load_failed_keywords(failed_keywords_file: str) -> set[str]:
    """실패한 검색어 목록을 로드합니다."""
    if not os.path.exists(failed_keywords_file): return set()
    with open(failed_keywords_file, 'r', encoding='utf-8') as f:
        failed_keywords = {line.strip() for line in f if line.strip()}
    logger.info(f"실패한 검색어 {len(failed_keywords)}개를 로드했습니다.")
    return failed_keywords


def save_failed_keyword(failed_keywords_file: str, keyword: str):
    """실패한 검색어를 파일에 기록합니다. (스레드 안전)"""
    with file_write_lock:
        os.makedirs(os.path.dirname(failed_keywords_file), exist_ok=True)
        with open(failed_keywords_file, 'a', encoding='utf-8') as f:
            f.write(keyword + '\n')
        logger.info(f"실패한 검색어 기록: {keyword}")


def extract_place_id_from_url(url: str) -> str | None:
    """URL에서 업체 ID를 추출합니다."""
    match = re.search(r"/place/(\d+)", url)
    return match.group(1) if match else None


def process_restaurant(driver: webdriver.Chrome, wait: WebDriverWait, restaurant_info: dict[str, any], crawled_place_ids: set, search_keyword_to_place_id: dict, failed_keywords: set, lock: threading.Lock) -> tuple[dict[str, any] | None, bool]:
    """개별 레스토랑 정보를 처리합니다."""
    title = restaurant_info.get("title", "").replace("&amp;", " ")
    road_address = restaurant_info.get("roadAddress", "")
    if not title or not road_address:
        logger.warning("제목 또는 주소가 비어있어 건너뜁니다.")
        return None, False

    short_address = " ".join(road_address.split()[:3])
    search_keyword = f"{title} {short_address}"
    
    if search_keyword in failed_keywords or search_keyword in search_keyword_to_place_id:
        logger.info(f"이미 처리된 검색어입니다: {search_keyword}. 건너뜁니다.")
        return None, False
    
    logger.info("=" * 80)
    logger.info(f"{title} ({search_keyword}) 크롤링 시작")

    try:
        driver.get(f"{config.base_url}{quote(search_keyword)}")
        if not switch_to_entry_iframe(driver, WebDriverWait(driver, 5)):
            logger.warning(f"Iframe을 찾지 못해 {title}을(를) 건너뜁니다.")
            return None, True

        place_id = extract_place_id_from_url(driver.current_url)
        if not place_id:
            logger.warning("URL에서 업체 ID를 찾지 못했습니다.")
            return None, True
        
        if place_id in crawled_place_ids:
            logger.info(f"이미 크롤링된 업체입니다 (ID: {place_id}). 건너뜁니다.")
            with lock:
                search_keyword_to_place_id[search_keyword] = place_id
            return None, False

        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.tab_container_selector)))
        
        menu_data = get_menu_data(driver, wait)
        review_data = get_review_data(driver, wait)
        description_data = get_info_description(driver, wait)

        crawled_data = {
            "place_id": place_id, "search_keyword": search_keyword, **restaurant_info,
            "menus": menu_data or [], "reviews": review_data or [], "description": description_data or "",
        }
        logger.info(f"수집 결과 - 메뉴: {len(menu_data or [])}개, 리뷰: {len(review_data or [])}개, 업체소개: {'있음' if description_data else '없음'}")
        return crawled_data, False

    except Exception as e:
        logger.error(f"'{title}' 크롤링 중 에러 발생: {e}")
        return None, True
    finally:
        logger.info("=" * 80)


def append_record_to_output(record: dict, output_dir: str):
    """크롤링된 데이터를 part 파일에 추가합니다. (스레드 안전)"""
    with file_write_lock:
        os.makedirs(output_dir, exist_ok=True)
        part_files = [f for f in os.listdir(output_dir) if f.startswith('part-') and f.endswith('.jsonl')]
        part_num = -1
        
        if not part_files:
            part_num = 0
        else:
            part_num = max(int(f.split('-')[1].split('.')[0]) for f in part_files)
            current_file_path = os.path.join(output_dir, f"part-{part_num:05d}.jsonl")
            if os.path.exists(current_file_path):
                with open(current_file_path, 'r', encoding='utf-8') as f:
                    if sum(1 for _ in f) >= MAX_RECORDS_PER_FILE:
                        part_num += 1
        
        output_file = os.path.join(output_dir, f"part-{part_num:05d}.jsonl")
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def crawler_worker(q: queue.Queue, crawled_place_ids: set, search_keyword_to_place_id: dict, failed_keywords: set):
    """큐에서 작업을 가져와 크롤링을 수행하는 워커 함수입니다."""
    driver = get_driver()
    wait = WebDriverWait(driver, config.default_wait_time)
    
    while not q.empty():
        try:
            restaurant_info = q.get_nowait()
        except queue.Empty:
            break

        crawled_data, should_record_failure = process_restaurant(
            driver, wait, restaurant_info,
            crawled_place_ids, search_keyword_to_place_id, failed_keywords, shared_data_lock
        )

        title = restaurant_info.get("title", "").replace("&amp;", " ")
        road_address = restaurant_info.get("roadAddress", "")
        search_keyword = f"{title} {" ".join(road_address.split()[:3])}"

        if should_record_failure:
            save_failed_keyword(FAILED_QUERIES_FILE, search_keyword)
            with shared_data_lock:
                failed_keywords.add(search_keyword)
        
        if crawled_data:
            append_record_to_output(crawled_data, OUTPUT_DIR)
            with shared_data_lock:
                place_id = crawled_data['place_id']
                crawled_place_ids.add(place_id)
                search_keyword_to_place_id[crawled_data['search_keyword']] = place_id
        
        q.task_done()

    driver.quit()
    logger.info("워커 종료.")


def main_concurrent():
    """동시성 크롤링을 설정하고 실행합니다."""
    logger.info(f"크롤링 시작 (최대 워커 수: {config.max_workers})")
    
    crawled_place_ids, search_keyword_to_place_id = load_existing_crawled_data(OUTPUT_DIR)
    failed_keywords = load_failed_keywords(FAILED_QUERIES_FILE)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    random.shuffle(lines)
    logger.info(f"총 {len(lines)}개 레스토랑 데이터 로드 및 셔플 완료")
    
    work_queue = queue.Queue()
    for line in lines:
        try:
            work_queue.put(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue
    
    logger.info(f"총 {work_queue.qsize()}개의 작업을 큐에 추가했습니다.")

    threads = []
    for _ in range(config.max_workers):
        thread = threading.Thread(
            target=crawler_worker,
            args=(work_queue, crawled_place_ids, search_keyword_to_place_id, failed_keywords),
            daemon=True
        )
        threads.append(thread)
        thread.start()

    # 모든 작업이 완료될 때까지 대기
    work_queue.join()
    logger.info("모든 작업이 완료되었습니다. 워커 스레드 종료를 기다립니다...")

    # 모든 스레드가 정상적으로 종료될 때까지 대기 (선택적 타임아웃)
    for thread in threads:
        thread.join(timeout=30)

    logger.info("크롤링 프로세스 완료.")


if __name__ == "__main__":
    main_concurrent()