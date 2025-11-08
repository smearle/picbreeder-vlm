/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.gui;

import client.*;
import client.evolution.*;
import client.renderers.*;
import java.awt.event.*;
import javax.swing.*;
import java.util.*;
import client.server.ServerConnectionInstance;
import client.server.ServerException;
import client.renderers.Renderer;
import client.renderers.RenderingAlgorithm;

// many apologies for the hacking in this class. it needs some major revising

public class EvolutionController extends JPanel implements ActionListener, TaskListener {
	private final PopulationPanel population;
	private final Collection <JComponent> controls;
	private final JComponent back, forward, redo, save;
	private boolean redoEnabled, saveEnabled;
	private Renderer renderer;
	private final List <Renderer> possibleRenderers;
	private final JPanel simpleEvoPanel, advancedEvoPanel, evoSubPanel;
	private final JCheckBox colorCheckBox;
	
	private final RenderingAlgorithm selectedAlgorithm = new client.renderers.algorithms.RowBased();
	private final RenderingAlgorithm deselectedAlgorithm = new client.renderers.algorithms.LowQualityTest();
	
	private static final Class [] NullClassArray = {};
	private static final Object [] NullObjectArray = {};
	
	private boolean registerRequired = false;
	private boolean anonymousPublish = false;
	private boolean canPublish = false;
	
	// for my own testing... ignore em
	long renderStartTime, renderEndTime;
	
	private JComboBox subnetOptions = null;
	
	public EvolutionController(PopulationPanel p) {
		super();

		client.renderers.RenderingAlgorithmInstance.set(selectedAlgorithm);
		
		population = p;
		controls = new LinkedList <JComponent> ();
		
		simpleEvoPanel = new JPanel();
		advancedEvoPanel = new JPanel();
		evoSubPanel = new JPanel();
		
		evoSubPanel.setBorder(BorderFactory.createTitledBorder("Controls"));
		evoSubPanel.setLayout(new java.awt.BorderLayout());
		simpleEvoPanel.setLayout(new java.awt.FlowLayout());
		advancedEvoPanel.setLayout(new java.awt.FlowLayout());

		possibleRenderers = new LinkedList <Renderer> ();
		if(CoreSpecificRenderer.isSupported())
			possibleRenderers.add(new CoreSpecificRenderer(this));
		possibleRenderers.add(new SingleRenderer(this));
		possibleRenderers.add(new ParallelRenderer(this));
		renderer = possibleRenderers.get(0);
		
		
		createGeneralControl("Quit",'\0',"toolbarButtonGraphics/general/Stop16.gif", "Aborts evolution and brings you back to the website.", false);
		//createAdvancedControl("Restart", '\0', "toolbarButtonGraphics/navigation/Home16.gif", "Start over from the picture you branched off of.");
		back = createAdvancedControl("Back", 'B', "toolbarButtonGraphics/navigation/Back16.gif", "Go back one generation (if possible)");
		forward = createAdvancedControl("Forward", 'F', "toolbarButtonGraphics/navigation/Forward16.gif", "Go forward one generaion (if possible)");
		redo = createAdvancedControl("Redo", 'R', "toolbarButtonGraphics/general/Redo16.gif", "Respawns this generation with the last pictures picked");
		createGeneralControl("Evolve", 'E', "toolbarButtonGraphics/media/Play16.gif", "Creates new images by evolving the images you have currently selected.");
		save = createAdvancedControl("Save", 'S', "toolbarButtonGraphics/general/Save16.gif", "Remembers your current stage of evolution so that you may resume from here later.");
		createGeneralControl("Publish", 'P', "toolbarButtonGraphics/general/SaveAll16.gif", "Stop evolving and publish the selected image for the world to see.");
		
		back.setEnabled(false);
		forward.setEnabled(false);
		redo.setEnabled(true);
		save.setEnabled(true);
		redoEnabled = true;
		saveEnabled = true;
		
		JComponent blah = createControlToggle();
		colorCheckBox = createColorToggle(); 
		blah.add(colorCheckBox);

		evoSubPanel.add(blah, java.awt.BorderLayout.NORTH);
		evoSubPanel.add(simpleEvoPanel, java.awt.BorderLayout.CENTER);
		
		this.setLayout(new java.awt.BorderLayout());
		
		this.add(createMenuBar(), java.awt.BorderLayout.NORTH);
		this.add(evoSubPanel, java.awt.BorderLayout.CENTER);
		this.add(createGuidancePanel(), java.awt.BorderLayout.SOUTH);
		
		try {			
			updatePopulation();
		}
		catch(ServerException e) {
			// can't load!
			JOptionPane.showMessageDialog(MainComponentInstance.get(), e.getMessage(), e.getClass().getSimpleName(), JOptionPane.ERROR_MESSAGE);
			quit();
		}
	}
	
	private JComponent createGeneralControl(String name, char hotkey, String icon, String tooltip) {
		return createGeneralControl(name, hotkey, icon, tooltip, true);
	}
	
	private JComponent createGeneralControl(String name, char hotkey, String icon, String tooltip, boolean disable) {
		JComponent control = createControl(name, hotkey, icon, tooltip, disable);
		advancedEvoPanel.add(control);
		simpleEvoPanel.add(createControl(name, hotkey, icon, tooltip, disable));
		return control;
	}
	
	private JComponent createAdvancedControl(String name, char hotkey, String icon, String tooltip) {
		JComponent control = createControl(name, hotkey, icon, tooltip, true);
		advancedEvoPanel.add(control);
		return control;
	}
	
	private JComponent createControl(String name, char hotkey, String icon, String tooltip, boolean disable) {
		JButton b = new JButton(name);
		b.setToolTipText(tooltip);
		b.addActionListener(this);
		b.setVerticalTextPosition(AbstractButton.BOTTOM);
		b.setHorizontalTextPosition(AbstractButton.CENTER);
		
		if(icon != null)
			b.setIcon(new ImageIcon(ImageDatabaseInstance.get().getImage(icon)));
		
		if(hotkey > '\0')
			b.setMnemonic(hotkey);
		
		if(disable)
			controls.add(b);
		return b;
	}
	
	private JComponent createControlToggle() {
		JPanel p = new JPanel();
		p.setLayout(new java.awt.FlowLayout());
		
		ButtonGroup group = new ButtonGroup();
		
		JRadioButton simpleControls = new JRadioButton("Basic");
		simpleControls.setToolTipText("Enables only the controls that are required for basic evolution.");
		simpleControls.setSelected(true);
		simpleControls.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				evoSubPanel.remove(advancedEvoPanel);
				evoSubPanel.add(simpleEvoPanel, java.awt.BorderLayout.CENTER);
				evoSubPanel.revalidate();
				repaint();
			}
		});
		group.add(simpleControls);
		p.add(simpleControls);

		JRadioButton advancedControls = new JRadioButton("Advanced");
		advancedControls.setToolTipText("Enables more advanced controls to guide evolution more specifically.");
		advancedControls.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				evoSubPanel.remove(simpleEvoPanel);
				evoSubPanel.add(advancedEvoPanel, java.awt.BorderLayout.CENTER);
				evoSubPanel.revalidate();
				repaint();
			}
		});
		group.add(advancedControls);
		p.add(advancedControls);
		
		return p;
	}

	private JCheckBox createColorToggle() {
		final JCheckBox display = new JCheckBox("Color");
		
		display.setToolTipText("Renders images in color.");
		display.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if(display.isSelected()) {
					subnetOptions.setEnabled(true);
					subnetOptions.setSelectedItem(subnetOptions.getSelectedItem());
					try {
						SeriesInstance.get().getCurrentGeneration().setDominantPhenotype(1);
					}
					catch(Exception exc) {
						exc.printStackTrace();
					}
				}
				else {
					subnetOptions.setEnabled(false);
					client.evolution.GeneticFactoryInstance.get().setScheme("structure");

					try {
						SeriesInstance.get().getCurrentGeneration().setDominantPhenotype(0);
					}
					catch(Exception exc) {
						exc.printStackTrace();
					}
				}
				population.repaint();
			}
		});
		
		return display;
	}
	
	private JComponent createEvolveColorDropBox() {
		JPanel p = new JPanel();
		p.setLayout(new java.awt.FlowLayout());
		
		subnetOptions = new SubnetComboBox();
		//options.setPreferredSize(new java.awt.Dimension(160, 20));
		
		subnetOptions.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				String thingy = subnetOptions.getSelectedItem().toString().toLowerCase();
				client.evolution.GeneticFactoryInstance.get().setScheme(thingy);
			}
		});
		
		subnetOptions.setEnabled(false);
	

		JLabel label = new JLabel("Focus:");
		label.setLabelFor(subnetOptions);
		
		subnetOptions.setToolTipText("Hints to the applet which part of the image to evolve.");
		
		p.add(label);
		p.add(subnetOptions);
		return p;
	}
	
	private JMenuBar createMenuBar() {		
		JMenuBar menuBar = new JMenuBar();
		menuBar.add(createSystemMenu());
		//menuBar.add(createControlsMenu());
		menuBar.add(createRenderMenu());
		return menuBar;
	}
	
	private JMenu createSystemMenu() {
		JMenu menu = new JMenu("System");
		menu.setMnemonic('s');
		
		JMenuItem restart = new JMenuItem("Restart");
		restart.setMnemonic('R');
		restart.addActionListener(this);
		menu.add(restart);
		
		JMenuItem quit = new JMenuItem("Quit");
		quit.setMnemonic('Q');
		quit.addActionListener(this);
		
		menu.add(quit);
		
		return menu;
	}
	
	private JMenu createControlsMenu() {
		JMenu menu = new JMenu("Controls");
		menu.setMnemonic('C');
		
		ButtonGroup group = new ButtonGroup();
		
		JMenuItem simpleControls = new JRadioButtonMenuItem("Basic");
		simpleControls.setSelected(true);
		simpleControls.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				evoSubPanel.remove(advancedEvoPanel);
				evoSubPanel.add(simpleEvoPanel, java.awt.BorderLayout.CENTER);
				evoSubPanel.revalidate();
				repaint();
			}
		});
		group.add(simpleControls);
		menu.add(simpleControls);

		JMenuItem advancedControls = new JRadioButtonMenuItem("Advanced");
		advancedControls.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				evoSubPanel.remove(simpleEvoPanel);
				evoSubPanel.add(advancedEvoPanel, java.awt.BorderLayout.CENTER);
				evoSubPanel.revalidate();
				repaint();
			}
		});
		group.add(advancedControls);
		menu.add(advancedControls);
		
		return menu;
	}
	
	private JMenu createRenderMenu() {
		JMenu menu = new JMenu("Render");
		menu.setMnemonic('R');
		ButtonGroup group = new ButtonGroup();
		
		ActionListener listener = new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				for(Renderer r : possibleRenderers)
					if(r.toString().equals(e.getActionCommand())) {
						renderer = r;
						break;
					}
			}
		};

		for(Renderer r : possibleRenderers) {
			JRadioButtonMenuItem item = new JRadioButtonMenuItem(r.toString());
			item.addActionListener(listener);
			item.setToolTipText(r.getDescription());
			
			menu.add(item);
			group.add(item);
			
			if(r == renderer)
				item.setSelected(true);
		}
		
		menu.add(new JSeparator());
		
		JCheckBoxMenuItem box = new JCheckBoxMenuItem("High Quality Rendering", true);
		
		box.addItemListener(new ItemListener() {
			public void itemStateChanged(ItemEvent e) {
				if(e.getStateChange() == ItemEvent.SELECTED)
					client.renderers.RenderingAlgorithmInstance.set(selectedAlgorithm);
				else if(e.getStateChange() == ItemEvent.DESELECTED)
					client.renderers.RenderingAlgorithmInstance.set(deselectedAlgorithm);
			}
		});
		
		box.setToolTipText("Higher quality rendering takes more time.");
		menu.add(box);
		
		JMenuItem refresh = new JMenuItem("Refresh");
		refresh.addActionListener(this);
		refresh.setToolTipText("Redraws the current population of images.");
		refresh.setMnemonic('R');
		menu.add(refresh);
		
		return menu;
	}
	
	public JPanel createGuidancePanel() {
		JPanel p = new JPanel();
		p.setLayout(new java.awt.BorderLayout());
		p.add(createEvolveColorDropBox(), java.awt.BorderLayout.NORTH);
		
		JPanel p2 = new JPanel();
		p.setBorder(BorderFactory.createTitledBorder("Guidance"));
		p2.setLayout(new java.awt.FlowLayout());
		
		JSlider slider = new ExplorationSlider(); 
		JLabel exploit = new JLabel("Small Changes");
		JLabel explore = new JLabel("Big Changes");
		exploit.setLabelFor(slider);
		explore.setLabelFor(slider);
		
		p2.add(exploit);
		p2.add(slider);
		p2.add(explore);
		
		String tip = "Guides the system by setting what kinds of changes you expect to see.";
		slider.setToolTipText(tip);
		exploit.setToolTipText(tip);
		explore.setToolTipText(tip);
		
		p.add(p2, java.awt.BorderLayout.CENTER);
		return p;
	}
	
	public void updatePopulation() throws client.server.ServerException {
		Generation g = SeriesInstance.get().getCurrentGeneration();
		population.setGeneration(g);

		updateColorState();
		
		renderer.render(g.getIndividuals());
	}
	
	private void updateColorState() throws client.server.ServerException {
		Individual ind = SeriesInstance.get().getCurrentGeneration().getRepresentative();
		if(ind == null)
			ind = SeriesInstance.get().getCurrentGeneration().getIndividual(0);
		
		int colorMode = ind.getGenome().getDominantPhenotype();
		
		if((colorMode == 1) != (colorCheckBox.isSelected()))
			colorCheckBox.doClick();
		
		for(Individual i : SeriesInstance.get().getCurrentGeneration())
			i.getGenome().setDominantPhenotype(colorMode);
	}
	
	public void notifyTaskStarting() {
		renderStartTime = System.currentTimeMillis();
		
		for(JComponent comp : controls)
			comp.setEnabled(false);
	}
	
	public void notifyTaskFinished() {
		renderEndTime = System.currentTimeMillis();
		
		InformationPanel.setLastRenderTime(renderEndTime - renderStartTime);
		//System.out.println("Last Render Time: " + (renderEndTime - renderStartTime));
		
		for(JComponent comp : controls)
			comp.setEnabled(true);
		
		forward.setEnabled(SeriesInstance.get().canGoForward());
		back.setEnabled(SeriesInstance.get().canGoBack());
		redo.setEnabled(redoEnabled);
		save.setEnabled(saveEnabled);

		// try to gc in case we have been eating space.
		// this is a good downtime
		System.runFinalization();
		System.gc();
	}
	
	public void actionPerformed(ActionEvent event) {
		try {
			String command = event.getActionCommand().toLowerCase();
			getClass().getMethod(command, NullClassArray).invoke(this, NullObjectArray);
		}
		catch(java.lang.reflect.InvocationTargetException e2) {
			e2.printStackTrace();
			Exception e = (Exception) e2.getTargetException();
			JOptionPane.showMessageDialog(MainComponentInstance.get(), e.getMessage(), e.getClass().getSimpleName(), JOptionPane.ERROR_MESSAGE);
		}
		catch(Exception e) {
			// shouldn't happen!
			e.printStackTrace();
			JOptionPane.showMessageDialog(MainComponentInstance.get(), e.getMessage(), e.getClass().getSimpleName(), JOptionPane.ERROR_MESSAGE);
		}
	}
	
	public void restart() throws EvolutionException, ServerException {
		SeriesInstance.get().initializeFirstGeneration();
		updatePopulation();
	}
	
	public void evolve() {
		spawn();
	}
	
	public void spawn() {
		try {
			redoEnabled = true;
			saveEnabled = true;
			SeriesInstance.get().spawn();
			updatePopulation();
		}
		catch(Exception e) {
			e.printStackTrace();
			JOptionPane.showMessageDialog(MainComponentInstance.get(), e.getMessage(), e.getClass().getSimpleName(), JOptionPane.ERROR_MESSAGE);
		}
	}
	
	public void back() throws ServerException {
		if(SeriesInstance.get().canGoBack()) {
			try {
				SeriesInstance.get().goBack();
				updatePopulation();
				redoEnabled = false;
				saveEnabled = true;
			}
			catch(ServerException e) {
				SeriesInstance.get().goForward();
				throw e;
			}
		}
	}
	
	public void forward(){
		if(SeriesInstance.get().canGoForward()) {
			try {
				SeriesInstance.get().goForward();
				updatePopulation();
				redoEnabled = false;
				saveEnabled = true;
			}
			catch(ServerException e) {
				e.printStackTrace();
				// SHOULD NOT HAPPEN
			}
		}
	}
	
	// TODO cleaner code
	public void redo() throws ServerException {
		if(redoEnabled) {
			if(SeriesInstance.get().canGoBack()) {
				try {
					saveEnabled = true;
					SeriesInstance.get().goBack();
					SeriesInstance.get().spawn();
					updatePopulation();
				}
				catch(EvolutionException e) {
					// happens when no parents were selected in the previous generation
					// for now report it
					JOptionPane.showMessageDialog(MainComponentInstance.get(), "Fixed Bug: Redo not possible. Please report!", "Bug", JOptionPane.INFORMATION_MESSAGE);
					SeriesInstance.get().goForward();
				}
				catch(ServerException e) {
					// happens when spawn tries to spawn from a
					// previous generation which was not able to load.
					// the back method succeeds, the spawn doesn't.
					// to restore, we must go forward
					SeriesInstance.get().goForward();
				}
			}
			else {
				SeriesInstance.get().initializeFirstGeneration();
				updatePopulation();
			}
		}
	}
	
	public void save()
			throws client.server.ServerException, EvolutionException {
		//if(JOptionPane.showConfirmDialog(MainComponentInstance.get(), "Known Bug Workaround: Saving will close the applet. Are you sure you want to save now?", "Known Bug: Save Series", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE) == JOptionPane.NO_OPTION)
		//	return;
		
		saveToServer();
		if(registerRequired)
			((SessionEnd) MainComponentInstance.get()).register();
		//quit();
	}
	
	/**
	 * Attempts to log the user into the system. If the user decide to cancel the operation,
	 * the method returns <code>false</code> so that an error box will not pop up.
	 * 
	 * @return <code>true</code> if the user wants to continue, <code>false</code> otherwise.
	 * @throws client.server.LoginException The user information is bad
	 * @throws client.server.TimeoutException The server cannot be reached
	 */
	private boolean loginUser() throws client.server.LoginException, client.server.TimeoutException {
		LoginBox box = new LoginBox();
		boolean r = box.hasLoginInformation();
		boolean login= box.hasLoginInformation();
		boolean anonymous= box.publishAnonymous();
		
		try {
			if(anonymous)
				r = anonymousPublish = anonymous;
			else if(login) {
				client.server.ServerConnectionInstance.get().logInExistingUser(box.getUserName(), box.getEncodedPassword());
				((SessionEnd)MainComponentInstance.get()).authenticate(box.getUserName(), box.getRawPassword());
			}
			else
				r = registerRequired = box.requiresRegistration();
		}
		catch(client.server.LoginException e) {
			throw e;
		}
		catch(client.server.TimeoutException e) {
			throw e;
		}
		finally {
			box.dispose();
		}
		
		return r;
	}
	
	private void saveToServer()
			throws client.server.FatalException, client.server.TimeoutException, client.server.LoginException, EvolutionException {
		Individual rep = null;
		
		try {
			rep = SeriesInstance.get().getCurrentGeneration().getRepresentative();
		}
		catch(ServerException e) {
			// won't happen
			e.printStackTrace();
		}
		
		if(rep == null)
			throw new EvolutionException("You must select a representative.");
		
		try {
			if(!client.server.ServerConnectionInstance.get().isUserLoggedIn())
				if(!loginUser()) {
					canPublish = false;
					return;
				}
			
			canPublish = true;
			
			String name = client.server.ServerConnectionInstance.get().getSeriesName();
			SeriesInstance.get().setCurrentBranch(name);
		}
		catch(client.server.FatalException e) {
			// TODO handle and quit
			throw e;
		}
		
		// Timeout will be handled by the actionPerformed method
		// no it won't, i'm retarded... ignore me
		// yes it will, i'm twice as retarded!
		try {
			client.utilities.XML.store(SeriesInstance.get(), ServerConnectionInstance.get().getSaveStreamForSeries());
			client.utilities.XML.store(rep.getGenome(), ServerConnectionInstance.get().getSaveStreamForGenome());
			
			java.util.Map <String, Transferable> dataMap = DatabaseInstance.get().getAdditionList();
			
			String [] remove = {}, add = {};
			
			remove = DatabaseInstance.get().getRemovalList().toArray(remove);
			add = dataMap.keySet().toArray(add);
			
			for(String storage : add)
				client.utilities.XML.store(dataMap.get(storage), ServerConnectionInstance.get().getSaveStreamForStorage(storage));
	
			if(registerRequired) {
				if(remove.length > 0)
					throw new client.evolution.EvolutionException("Internal Registration Error. PLEASE REPORT THIS!");
				ServerConnectionInstance.get().saveAnonymously(add);
			}
			else
				ServerConnectionInstance.get().save(remove, add);
			
			SeriesInstance.get().notifySaveSuccessful();
		}
		// TODO better exception handling
		catch(java.io.IOException e) {
			throw new client.server.FatalException(e.getMessage());
		}
		
		saveEnabled = false;
		save.setEnabled(false);
		
	}
	
	
	public void publish() throws client.server.ServerException, EvolutionException {
		if(saveEnabled)
			saveToServer();
		
		if(canPublish) {
			if(registerRequired) {
				JOptionPane.showMessageDialog(MainComponentInstance.get(), "Please publish this image from your user panel after you confirm your account.", "Anonymous Publish", JOptionPane.INFORMATION_MESSAGE);
				((SessionEnd) MainComponentInstance.get()).register();
			}
			else if(anonymousPublish)
				((SessionEnd) MainComponentInstance.get()).publishanonymously();
			else
				((SessionEnd) MainComponentInstance.get()).publish();
		}
	}
	
	public void quit() {
		if(saveEnabled) {
			int res = JOptionPane.showConfirmDialog(MainComponentInstance.get(), "Quitting will destroy unsaved data. Are you sure you want to quit?", "Unsaved Data", JOptionPane.YES_NO_OPTION, JOptionPane.WARNING_MESSAGE);
			
			if(res != JOptionPane.YES_OPTION)
				return;
		}
		
		((SessionEnd) MainComponentInstance.get()).quit();	
	}
	
	public void refresh() {
		try {
			updatePopulation();
		}
		catch(ServerException e) {
			// won't happen
		}
	}
}
