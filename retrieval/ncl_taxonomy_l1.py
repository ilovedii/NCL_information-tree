CATEGORY_NAMES = ['編目政策與管理', '機讀編目格式', '編目法', '分類法', '主題法', '作者號及輔助區分號', '權威控制', '其他']

CATEGORY_PROFILES = {'編目政策與管理': {'description': '館內編目政策、作業流程、抄編與原編政策、合作編目、編目外包、委外人力、編目分工、書目品質、品質管理、館藏處理慣例、簡體字著錄、採購或編目系統導入等。',
             'example': '簽名書如何處理、館內如何訂定編目作業政策。'},
 '機讀編目格式': {'description': 'MARC、CMARC、MARC 21、ISO 2709 等書目與權威格式的欄位功能、欄位結構、指標、分欄、代碼、固定長欄位、變長欄位、館藏段、資料交換、系統編碼與格式呈現。',
            'example': '欄位 490 與 830 的關係、欄位 300 用途、LDR/06 代碼、欄位中的 $a 與 $b 如何使用。'},
 '編目法': {'description': 'RDA、AACR、AACR2、中國編目規則、資源描述與檢索、書目著錄原則（包括題名、著者、版本、出版項、出版者、出版年、稽核項、載體、集叢、附註、ISSN、標準號碼、識別碼等資料應如何判定或著錄）、權威著錄原則、創作者、貢獻者、實體、屬性、關係、RDA '
                        '元素、屬性元素、關係元素、主要款目、附加款目、檢索款目、款目擇定。即使問題或答案提到特定 MARC 欄位，若核心是書目資料應如何著錄，仍歸此類。',
         'example': '出版者是否照錄、影印本缺出版資料如何著錄、ISSN 應依何項規則著錄。'},
 '分類法': {'description': '中國圖書分類法、中文圖書分類法、DDC、LCC、分類號、類目歸屬、分類表結構、分類表使用、複分、通用複分、專類複分、仿分、總論複分、標準複分、類號組配、主題應分入哪一類，以及分類號的修訂或適用範圍。',
         'example': '某書應分哪一類、類號是否衝突、如何使用時代表或地區複分。'},
 '主題法': {'description': '中文圖書標題表、中文主題詞表、LCSH、主題分析、主題詞、標題、自由詞、參考類號、參照、參見、主題標引、詞彙控制、詞間關係、詞彙組配、主題詞表與標引詞目使用、主題詞增訂。',
         'example': '某概念應採哪個主題詞、主題詞間的參見關係、地方名是否可作主題詞。'},
 '作者號及輔助區分號': {'description': '作者號、著者號、Cutter no.、克特號、作品號、種次號、冊次號、續編號、作者區分號及其他索書號中的輔助區分資訊。',
               'example': '團體作者如何取作者號、冊次號如何標示、特定作者名稱應取何種著者號。'},
 '權威控制': {'description': '個人名稱、團體名稱、會議名稱、題名、主題等權威標目的標準形式；異名、參照、標目關係、劃一題名、權威紀錄與權威檔。',
          'example': '人名應採何種標準形式、異名如何建立參照、權威紀錄如何處理。'},
 '其他': {'description': '不屬於上述七類的問題，例如一般圖書館術語定義、國圖組織或行政資訊、ISBN 申請、Z39.50 連線設定、非編目專業的系統選購建議、學理討論或無法明確歸類的問題。',
        'example': '一般圖書館術語、國圖行政資訊、ISBN 申請、Z39.50 連線設定等。'}}


def category_profile_text(category_name):
    profile = CATEGORY_PROFILES[category_name]
    return (
        f"類別：{category_name}。"
        f"分類標準：{profile['description']}"
        f"例如：{profile['example']}"
    )


def category_standard_text():
    parts = ["【國家圖書館 Level 1 分類標準】"]
    for i, name in enumerate(CATEGORY_NAMES, start=1):
        profile = CATEGORY_PROFILES[name]
        parts.append(
            f"{i}. {name}\n"
            f"{profile['description']}\n"
            f"例如：{profile['example']}"
        )
    return "\n\n".join(parts)


def validate_l1_taxonomy():
    if len(CATEGORY_NAMES) != len(set(CATEGORY_NAMES)):
        raise ValueError("L1 類別名稱重複")
    missing = [name for name in CATEGORY_NAMES if name not in CATEGORY_PROFILES]
    if missing:
        raise ValueError(f"L1 缺少 profile：{missing}")
    extra = [name for name in CATEGORY_PROFILES if name not in CATEGORY_NAMES]
    if extra:
        raise ValueError(f"L1 profile 含未列入 CATEGORY_NAMES 的類別：{extra}")
    for name, profile in CATEGORY_PROFILES.items():
        for key in ("description", "example"):
            if not str(profile.get(key, "")).strip():
                raise ValueError(f"{name} 缺少 {key}")
    return {"l1_count": len(CATEGORY_NAMES)}


L1_TAXONOMY_VALIDATION = validate_l1_taxonomy()
