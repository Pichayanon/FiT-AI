# Firebase + Login Setup (FIT-AI)

คุณมี **GoogleService-Info.plist** แล้ว ทำตามลำดับด้านล่างได้เลย

---

## 1. เพิ่ม CLIENT_ID ใน plist (สำหรับ Google Sign-In)

plist ตอนนี้ยังไม่มี `CLIENT_ID` / `REVERSED_CLIENT_ID` ต้องให้ Firebase สร้าง OAuth client ให้ก่อน:

1. เปิด [Firebase Console](https://console.firebase.google.com/) → โปรเจกต์ **fit-ai-64704**
2. ไป **Authentication** → แท็บ **Sign-in method**
3. กด **Google** → เปิด **Enable** → ตั้ง Support email → **Save**
4. ไป **Project settings** (ไอคอนเฟือง) → ส่วน **Your apps** → เลือกแอป iOS (Bundle ID `th.ac.ku.cpe.frontend`)
5. **ดาวน์โหลด GoogleService-Info.plist ใหม่** จากหน้านี้ แล้ว**แทนที่**ไฟล์เดิมในโปรเจกต์ (หรือลากลงใน Xcode แทนที่ของเก่า)

ใน plist ตัวใหม่ควรมี key **CLIENT_ID** และ **REVERSED_CLIENT_ID** ถ้ายังไม่มี แปลว่า OAuth client ยังไม่ถูกสร้าง ให้ทำข้อ 3 ให้เรียบร้อยแล้วดาวน์โหลด plist ใหม่อีกครั้ง

---

## 2. เปิดใช้ Sign-in method อื่น (ถ้าจะใช้)

- **Apple**: ใน **Authentication** → **Sign-in method** → **Apple** → เปิด **Enable** → Save

---

## 3. Xcode – Sign in with Apple (ถ้าใช้ Apple)

1. เปิดโปรเจกต์ใน Xcode
2. เลือก target **frontend** → **Signing & Capabilities**
3. กด **+ Capability** → เลือก **Sign in with Apple**

---

## 4. Xcode – URL Scheme สำหรับ Google Sign-In

1. เปิด **GoogleService-Info.plist** (ตัวที่อัปเดตแล้วและมี REVERSED_CLIENT_ID)
2. หาค่า **REVERSED_CLIENT_ID** (รูปแบบประมาณ `com.googleusercontent.apps.xxxxxx`)
3. ใน Xcode: เลือก target **frontend** → แท็บ **Info** → **URL Types**
4. กด **+** เพิ่ม 1 รายการ:
   - **Identifier**: `com.google.signin`
   - **URL Schemes**: ใส่ค่า **REVERSED_CLIENT_ID** จาก plist (ค่าเดียวกับข้อ 2)
   - **Role**: Editor

---

## 5. Build และรัน

- กด Build (⌘B) แล้วรันบน Simulator หรือเครื่องจริง
- เปิดแอปจะเห็นหน้า Login → ลองกด **Continue with Google** หรือ **Sign in with Apple**

ถ้า Google Sign-In ขึ้น error เรื่อง client ID / URL scheme ให้ตรวจสอบว่า plist มี CLIENT_ID และ REVERSED_CLIENT_ID และว่า URL Schemes ใน Info ตรงกับ REVERSED_CLIENT_ID
