import { useState } from "react";
import FormSwitch from "./FormSwitch";

export type AlertSettings = {
  soundOnWarning: boolean;
  soundOnDanger: boolean;
  flashing: boolean;
}

function DisplaySettings({ settings, setSettings }: { settings: AlertSettings, setSettings: (settings: AlertSettings) => void }) {

    const setWarningSound = (checked: boolean) => {
        console.log("Warning sound: ", checked);
        setSettings({ ...settings, soundOnWarning: checked });
    }

    const setEmergencyBrakeSound = (checked: boolean) => {
        console.log("Emergency brake sound: ", checked);
        setSettings({ ...settings, soundOnDanger: checked });
    }

    const setDisplayFlashing = (checked: boolean) => {
        console.log("Display flashing: ", checked);
        setSettings({ ...settings, flashing: checked });
    }

    return (
        <>
            <button data-modal-target="settings-modal" data-modal-toggle="settings-modal" className="block text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 text-center" type="button">
                Settings
            </button>

            <div id="settings-modal" tabIndex={-1} aria-hidden="true" className="hidden overflow-y-auto overflow-x-hidden fixed top-0 right-0 left-0 z-50 justify-center items-center w-full md:inset-0 h-[calc(100%-1rem)] max-h-full">
                <div className="relative p-4 w-full max-w-xl max-h-full">
                    <div className="relative bg-white rounded-lg shadow-sm ">
                        <div className="flex items-center justify-between p-4 md:p-5 border-b rounded-t  border-gray-200">
                            <h3 className="text-xl font-semibold text-gray-900 ">
                                Settings
                            </h3>
                            <button type="button" className="end-2.5 text-gray-400 bg-transparent hover:bg-gray-200 hover:text-gray-900 rounded-lg text-sm w-8 h-8 ms-auto inline-flex justify-center items-center " data-modal-hide="settings-modal">
                                <svg className="w-3 h-3" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 14 14">
                                    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="m1 1 6 6m0 0 6 6M7 7l6-6M7 7l-6 6"/>
                                </svg>
                                <span className="sr-only">Close modal</span>
                            </button>
                        </div>
                        <div className="p-5">
                            <form className="p-5">
                                <ul className="">
                                    <li className="mb-5">
                                        <FormSwitch label="Warning Sound" checked={settings.soundOnWarning} onClick={setWarningSound}>
                                            Whenever a pedestrian is detected, a warning sound will be played
                                        </FormSwitch>
                                    </li>
                                    <li className="mb-5">
                                        <FormSwitch label="Emergency Brake Sound" checked={settings.soundOnDanger} onClick={setEmergencyBrakeSound}>
                                            Whenever the vehicle detects an imminent collision, an emergency brake sound will be played
                                        </FormSwitch>
                                    </li>
                                    <li className="mb-5">
                                        <FormSwitch label="Display flashing" checked={settings.flashing} onClick={setDisplayFlashing}>
                                            When pedestrians are detected the screen will flash
                                        </FormSwitch>
                                    </li>
                                </ul>
                                <div className="mt-10 flex justify-around">
                                    <input className="bg-blue-500 text-white px-4 py-2 rounded" type="submit" value="Save Settings" />
                                    <input className="bg-gray-300 text-gray-700 px-4 py-2 rounded" type="reset" value="Cancel" data-modal-hide="settings-modal"/>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}

export default DisplaySettings;