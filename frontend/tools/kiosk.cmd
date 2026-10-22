@echo off
rem kiosk.cmd -- the open-day stand: the game full screen in Chrome, nothing else on it.
rem
rem   * ?kiosk=1 hides the links off the game (the scenarios, the map).
rem   * Its own Chrome profile, one per day: the leaderboard starts empty each morning and
rem     keeps the day's scores if Chrome is closed and opened again. Save it at the end of the
rem     day with "Save the board as a file" on the results screen.
rem   * Leave it with Alt+F4.
rem
rem Usage: double-click, or  kiosk.cmd [url]   (a local build: kiosk.cmd http://localhost:4173/game.html?kiosk=1)

setlocal
set "URL=%~1"
if "%URL%"=="" set "URL=https://aditya-raghunandan.github.io/MaritimeSAROptimisation/game.html?kiosk=1"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "DAY=%%d"
set "PROFILE=%LOCALAPPDATA%\survive-the-triangle-kiosk\%DAY%"

set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" (
  echo Chrome not found. In Edge: msedge --kiosk "%URL%" --edge-kiosk-type=fullscreen
  pause
  exit /b 1
)

start "" "%CHROME%" --kiosk --user-data-dir="%PROFILE%" --no-first-run --no-default-browser-check --disable-session-crashed-bubble --disable-features=TranslateUI --autoplay-policy=no-user-gesture-required "%URL%"
