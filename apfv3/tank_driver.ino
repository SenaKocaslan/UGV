/*
  tank_driver.ino — Paletli tank motor sürücüsü (BTS7960)
  ═══════════════════════════════════════════════════════════
  ROS2 cmd_vel_serial node'u şu formatta seri komut gönderir:

      L<sol_pwm> R<sag_pwm>\n
      Örnek: "L128 R-200\n"
      Değer: -255 … +255   (+ ileri, - geri)

  Baud: 115200

  BTS7960 sürüş mantığı:
    İleri  → RPWM = pwm,  LPWM = 0
    Geri   → RPWM = 0,   LPWM = pwm
    Dur    → RPWM = 0,   LPWM = 0   (serbest)
    R_EN ve L_EN her zaman HIGH (enable)

 * BAĞLANTI ŞEMASI:
 * ┌──────────────────────────────────────────────┐
 * │  BTS7960 #1 — SOL MOTOR                      │
 * │  R_EN  → Mega Pin 22  (Digital)              │
 * │  L_EN  → Mega Pin 23  (Digital)              │
 * │  RPWM  → Mega Pin  2  (PWM ~ Timer3)         │
 * │  LPWM  → Mega Pin  3  (PWM ~ Timer3)         │
 * │  VCC   → Mega 5V                             │
 * │  GND   → Mega GND                            │
 * │  M+/M- → Sol motor uçları                   │
 * ├──────────────────────────────────────────────┤
 * │  BTS7960 #2 — SAĞ MOTOR                      │
 * │  R_EN  → Mega Pin 24  (Digital)              │
 * │  L_EN  → Mega Pin 25  (Digital)              │
 * │  RPWM  → Mega Pin  4  (PWM ~ Timer3)         │
 * │  LPWM  → Mega Pin  5  (PWM ~ Timer3)         │
 * │  VCC   → Mega 5V                             │
 * │  GND   → Mega GND                            │
 * │  M+/M- → Sağ motor uçları                   │
 * └──────────────────────────────────────────────┘
*/

// ─── Pin tanımları ─────────────────────────────────────────────────────────
// Sol motor — BTS7960 #1
#define L_R_EN   22
#define L_L_EN   23
#define L_RPWM    2
#define L_LPWM    3

// Sağ motor — BTS7960 #2
#define R_R_EN   24
#define R_L_EN   25
#define R_RPWM    4
#define R_LPWM    5

// ─── Güvenlik watchdog (ms) ────────────────────────────────────────────────
#define WATCHDOG_MS  500

// ─── Global değişkenler ────────────────────────────────────────────────────
static String   g_buf;
static uint32_t g_last_cmd_ms   = 0;
static bool     g_watchdog_active = false;

// ─── BTS7960 motor sürme ──────────────────────────────────────────────────
// pwm_val: -255 … +255  (+ ileri, - geri)
void driveMotor(uint8_t rpwm_pin, uint8_t lpwm_pin, int pwm_val)
{
  pwm_val = constrain(pwm_val, -255, 255);

  if (pwm_val > 0) {
    analogWrite(rpwm_pin, pwm_val);
    analogWrite(lpwm_pin, 0);
  } else if (pwm_val < 0) {
    analogWrite(rpwm_pin, 0);
    analogWrite(lpwm_pin, -pwm_val);
  } else {
    analogWrite(rpwm_pin, 0);
    analogWrite(lpwm_pin, 0);
  }
}

void stopAll()
{
  driveMotor(L_RPWM, L_LPWM, 0);
  driveMotor(R_RPWM, R_LPWM, 0);
}

// ─── Komut ayrıştırma ─────────────────────────────────────────────────────
// Beklenen format: "L<int> R<int>"
// Örnek: "L128 R-200"
void parseCommand(const String &line)
{
  int l_pos = line.indexOf('L');
  int r_pos = line.indexOf('R');
  if (l_pos < 0 || r_pos < 0 || r_pos <= l_pos) return;

  int left_pwm  = line.substring(l_pos + 1, r_pos).toInt();
  int right_pwm = line.substring(r_pos + 1).toInt();

  driveMotor(L_RPWM, L_LPWM, left_pwm);
  driveMotor(R_RPWM, R_LPWM, right_pwm);

  g_last_cmd_ms     = millis();
  g_watchdog_active = true;
}

// ─── Setup ────────────────────────────────────────────────────────────────
void setup()
{
  // Enable pinleri HIGH (BTS7960 aktif)
  pinMode(L_R_EN, OUTPUT); digitalWrite(L_R_EN, HIGH);
  pinMode(L_L_EN, OUTPUT); digitalWrite(L_L_EN, HIGH);
  pinMode(R_R_EN, OUTPUT); digitalWrite(R_R_EN, HIGH);
  pinMode(R_L_EN, OUTPUT); digitalWrite(R_L_EN, HIGH);

  // PWM pinleri çıkış
  pinMode(L_RPWM, OUTPUT);
  pinMode(L_LPWM, OUTPUT);
  pinMode(R_RPWM, OUTPUT);
  pinMode(R_LPWM, OUTPUT);

  stopAll();

  Serial.begin(115200);
  g_buf.reserve(32);

  Serial.println("TANK_DRIVER_READY");
}

// ─── Loop ─────────────────────────────────────────────────────────────────
void loop()
{
  // Seri veri oku
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      g_buf.trim();
      if (g_buf.length() > 0) {
        parseCommand(g_buf);
      }
      g_buf = "";
    } else if (g_buf.length() < 31) {
      g_buf += c;
    }
  }

  // Watchdog: komut kesilirse dur
  if (g_watchdog_active && (millis() - g_last_cmd_ms > WATCHDOG_MS)) {
    stopAll();
    g_watchdog_active = false;
    Serial.println("WATCHDOG_STOP");
  }
}
