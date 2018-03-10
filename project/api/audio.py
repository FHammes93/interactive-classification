import pyttsx
import speech_recognition as sr


class Audio:

    def __init__(self):
        pass

    # Speaker (text to speech)
    def output(self, text):

        # initialize
        engine = pyttsx.init()

        # set properties
        engine.setProperty('rate', 175)
        engine.setProperty('voice', 'english-us')

        # speak
        engine.say(text)
        engine.runAndWait()
        print(text)

    # Extracts a text from the user via speech or keyboard
    def input(self, question, speech=True):

        # initialize
        r = sr.Recognizer()
        m = sr.Microphone()
        judge_classify = ""

        self.output(question)

        # Recognition, keep asking until rightly recognized
        while speech is True:
            try:
                with m as source:
                    r.adjust_for_ambient_noise(source)
                    audio = r.listen(source)
                self.output("Got it! Now to recognize it...")
                try:
                    # Recognize speech using Google Speech Recognition
                    judge_classify = r.recognize_google(audio)

                    # Show and speak the result
                    sentence = 'You said ' + judge_classify
                    self.output(sentence)

                    # Waiting for judging
                    self.output('Is it right?')

                    # Confirm with speech
                    with m as source:
                        r.adjust_for_ambient_noise(source)
                        audio = r.listen(source)
                    self.output("Got it! Now to recognize it...")
                    # Recognize speech using Google Speech Recognition
                    judge_verify = r.recognize_google(audio)

                    # Finish when rightly recognized
                    if judge_verify == 'yes':
                        break
                # by unknown value error
                except sr.UnknownValueError:
                    self.output("Oops! Didn't catch that. Please say it again.")
                # by request error
                except sr.RequestError as e:
                    self.output("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
            except KeyboardInterrupt:
                pass
            # Waiting for another speech if wrongly recognized
            self.output('Please say it again')

        if speech is False:
            judge_classify = raw_input("Answer: ")

        return judge_classify


